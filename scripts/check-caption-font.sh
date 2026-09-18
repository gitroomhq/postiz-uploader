#!/bin/sh
# Build-time check: libass must resolve the default caption face to the real font.
# When fontconfig is misconfigured libass silently falls back to whatever it finds,
# and the first sign would be ugly captions in production.
set -eu
dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT
cat > "$dir/t.ass" <<'ASS'
[Script Info]
ScriptType: v4.00+
PlayResX: 320
PlayResY: 240

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Montserrat,24,&HFFFFFF&,&HFFFFFF&,&H000000&,&H80000000&,-1,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,font check
ASS
log=$(ffmpeg -hide_banner -loglevel verbose -f lavfi -i color=c=black:s=320x240:d=0.2 \
  -vf "ass=filename=$dir/t.ass" -f null - 2>&1)
echo "$log" | grep -i "fontselect" || true
echo "$log" | grep -i "fontselect" | grep -q "Montserrat-Bold\.otf" || {
  echo "libass did not resolve Montserrat bold to Montserrat-Bold.otf" >&2
  exit 1
}
