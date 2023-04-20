from PIL import Image
import os


for q in range(1, 9):
    for i in range(1, 25): 
        # Open the JPEG image
        jpeg_path = '{}kodim{:02d}.jpeg'.format(q, i)
        with Image.open(jpeg_path) as img:
            # Get the size of the JPEG image
            size = img.size
            
            # Convert the image to PNG format
            png_path = '{}kodim{:02d}.png'.format(q, i)
            img.save(png_path, 'PNG')
            
            # Open the PNG image and resize it to the original size of the JPEG image
            with Image.open(png_path) as png_img:
                png_img = png_img.resize(size)
                png_img.save(png_path)