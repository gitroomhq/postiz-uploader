"""Still images with Pillow. GIFs are never re-encoded."""

from __future__ import annotations

import os
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from postiz_uploader import errors
from postiz_uploader.config import Settings
from postiz_uploader.errors import JobError
from postiz_uploader.planner import clamp_dimensions
from postiz_uploader.schema import Job

# Pillow's own bomb guard would raise before our check; we enforce our own cap
Image.MAX_IMAGE_PIXELS = None

_FORMAT_TO_TYPE = {"JPEG": "image/jpeg", "MPO": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
_SAVE_FORMAT = {"JPEG": "JPEG", "MPO": "JPEG", "PNG": "PNG", "WEBP": "WEBP"}
ORIENTATION_TAG = 0x0112


@dataclass
class ImageOutcome:
    status: str  # completed | unchanged
    actions: tuple[str, ...]
    source: dict
    output_path: str | None
    output: dict | None


def process_image(job: Job, input_path: str, workdir: str, settings: Settings) -> ImageOutcome:
    try:
        img = Image.open(input_path)
    except (UnidentifiedImageError, OSError) as err:
        raise JobError(errors.UNSUPPORTED_INPUT, f"cannot read image: {err.__class__.__name__}") from err

    with img:
        fmt = img.format or ""
        width, height = img.size
        animated = bool(getattr(img, "is_animated", False))
        source = {
            "format": fmt.lower(),
            "width": width,
            "height": height,
            "animated": animated,
            "bytes": os.path.getsize(input_path),
        }

        if width * height > settings.image_max_pixels:
            raise JobError(
                errors.UNSUPPORTED_INPUT,
                f"image has {width * height} pixels, cap is {settings.image_max_pixels}",
            )

        # never re-encode a GIF, and never flatten an animation (animated WebP) to one frame
        if fmt == "GIF" or animated:
            return ImageOutcome("unchanged", (), source, None, None)

        if fmt not in _FORMAT_TO_TYPE:
            raise JobError(errors.UNSUPPORTED_INPUT, f"unsupported image format {fmt or 'unknown'}")

        expected_type = _FORMAT_TO_TYPE[fmt]
        if job.rules.image.keep_format and job.output.content_type != expected_type:
            raise JobError(
                errors.INVALID_JOB,
                f"output.content_type is {job.output.content_type} but keep_format requires {expected_type}",
            )

        orientation = img.getexif().get(ORIENTATION_TAG, 1) or 1
        needs_orient = orientation != 1
        icc = img.info.get("icc_profile")

        # Load now so exif_transpose works on decoded data and the file can be released.
        # A truncated or corrupt file passes the lazy header parse and fails here.
        try:
            img.load()
            work = ImageOps.exif_transpose(img) if needs_orient else img
        except (OSError, ValueError, SyntaxError) as err:
            raise JobError(errors.UNSUPPORTED_INPUT, f"cannot decode image: {err}") from err
        width, height = work.size
        # rotation may have swapped the sides; report post-orientation dimensions
        source["width"], source["height"] = width, height

        target_w, target_h = clamp_dimensions(width, height, job.rules, even=False)
        needs_scale = (target_w, target_h) != (width, height)

        if not needs_orient and not needs_scale:
            return ImageOutcome("unchanged", (), source, None, None)

        actions: list[str] = []
        if needs_scale:
            actions.append("scale")
            try:
                work = work.resize((target_w, target_h), Image.LANCZOS)
            except (OSError, ValueError) as err:
                raise JobError(errors.ENCODE_FAILED, f"Pillow resize failed: {err}") from err
        if needs_orient:
            actions.append("orient")

        save_format = _SAVE_FORMAT[fmt]
        output_path = os.path.join(workdir, f"output.{save_format.lower()}")
        params: dict = {}
        if icc:
            params["icc_profile"] = icc
        if save_format == "JPEG":
            if work.mode not in ("RGB", "L"):
                # a CMYK profile must not be embedded in the RGB output
                params.pop("icc_profile", None)
                work = work.convert("RGB")
            params.update(quality=job.rules.image.jpeg_quality, optimize=True, progressive=True)
        elif save_format == "PNG":
            params.update(optimize=True)
        elif save_format == "WEBP":
            params.update(quality=job.rules.image.jpeg_quality, method=4)

        try:
            work.save(output_path, format=save_format, **params)
        except (OSError, ValueError) as err:
            raise JobError(errors.ENCODE_FAILED, f"Pillow save failed: {err}") from err

        output = {
            "width": target_w,
            "height": target_h,
            "bytes": os.path.getsize(output_path),
            "content_type": expected_type,
        }
        return ImageOutcome("completed", tuple(actions), source, output_path, output)
