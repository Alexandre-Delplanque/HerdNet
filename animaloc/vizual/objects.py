import PIL

from PIL import Image, ImageDraw, ImageFont

__all__ = ['draw_points', 'draw_boxes', 'draw_text']

def draw_points(
    image: PIL.Image.Image, 
    points: list, 
    color: str = 'red',
    size: int = 4
    ) -> PIL.Image.Image:

    draw = ImageDraw.Draw(image)

    for p in points:
        e = [
            p[1] - (size // 2), p[0] - (size // 2),
            p[1] + (size // 2), p[0] + (size // 2)
            ]

        draw.ellipse(e, fill=color, outline='black')
  
    return image

def draw_boxes(
    image: PIL.Image.Image, 
    boxes: list, 
    color: str = 'red',
    width: int = 1
    ) -> PIL.Image.Image:

    draw = ImageDraw.Draw(image)

    for b in boxes:
        draw.rectangle(b, fill=None, outline=color, width=width)
  
    return image

def draw_text(
    image: PIL.Image.Image,
    text: str,
    position: tuple,
    font_size: int = 20,
    )-> PIL.Image.Image:

    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("segoeui.ttf", size=font_size)

    l, t, r, b = draw.textbbox(position, text, font=font)
    draw.rectangle((l-5, t-5, r+5, b+5), fill='white')
    draw.text(position, text, font=font, fill='black')

    return image
