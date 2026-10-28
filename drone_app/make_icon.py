"""Draws the Drone App icon to the given .ico path."""
import sys

from PIL import Image, ImageDraw


def draw(size: int = 256) -> Image.Image:
    image = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    pen = ImageDraw.Draw(image)
    pen.rounded_rectangle((4, 4, size - 4, size - 4), radius=40, fill="white", outline="black", width=10)
    a, b = int(size * 0.3), int(size * 0.7)
    pen.line((a, a, b, b), fill="black", width=18)
    pen.line((a, b, b, a), fill="black", width=18)
    r = int(size * 0.13)
    for x, y in ((a, a), (b, a), (a, b), (b, b)):
        pen.ellipse((x - r, y - r, x + r, y + r), outline="black", width=10, fill="white")
    c, d = size // 2, int(size * 0.09)
    pen.ellipse((c - d, c - d, c + d, c + d), fill="#c40000")
    return image


if __name__ == "__main__":
    draw().save(sys.argv[1], sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
