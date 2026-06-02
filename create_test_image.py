"""
Create a sample test image for API testing
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_test_image():
    """Create a sample image that resembles an eye for testing"""
    
    # Create image
    width, height = 640, 480
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Draw background (skin tone)
    draw.rectangle([0, 0, width, height], fill=(220, 180, 150))
    
    # Draw left eye (circular shape)
    eye_x, eye_y = 200, 180
    eye_size = 80
    draw.ellipse([eye_x - eye_size, eye_y - eye_size, eye_x + eye_size, eye_y + eye_size], 
                 fill=(255, 255, 255), outline=(0, 0, 0), width=3)
    
    # Draw iris
    iris_size = 30
    draw.ellipse([eye_x - iris_size, eye_y - iris_size, eye_x + iris_size, eye_y + iris_size],
                 fill=(101, 67, 33), outline=(0, 0, 0), width=2)
    
    # Draw pupil
    pupil_size = 12
    draw.ellipse([eye_x - pupil_size, eye_y - pupil_size, eye_x + pupil_size, eye_y + pupil_size],
                 fill=(0, 0, 0), outline=(0, 0, 0), width=1)
    
    # Draw right eye
    eye_x2, eye_y2 = 440, 180
    draw.ellipse([eye_x2 - eye_size, eye_y2 - eye_size, eye_x2 + eye_size, eye_y2 + eye_size],
                 fill=(255, 255, 255), outline=(0, 0, 0), width=3)
    draw.ellipse([eye_x2 - iris_size, eye_y2 - iris_size, eye_x2 + iris_size, eye_y2 + iris_size],
                 fill=(101, 67, 33), outline=(0, 0, 0), width=2)
    draw.ellipse([eye_x2 - pupil_size, eye_y2 - pupil_size, eye_x2 + pupil_size, eye_y2 + pupil_size],
                 fill=(0, 0, 0), outline=(0, 0, 0), width=1)
    
    # Add some variation (redness, etc.) to make it more realistic
    pixels = np.array(image)
    # Add some red around edges to simulate conjunctivitis
    pixels[150:210, 120:180] = np.array([240, 100, 100])  # Red around left eye
    pixels[150:210, 360:420] = np.array([220, 150, 150])  # Light red around right eye
    
    image = Image.fromarray(pixels.astype('uint8'))
    
    # Add text
    try:
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), "YOLO v12 Test Image", fill=(0, 0, 0))
    except:
        pass
    
    # Save
    image.save("assets/test_image.jpg", "JPEG", quality=95)
    print(f"✓ Sample test image created: assets/test_image.jpg ({width}x{height})")
    return image

if __name__ == "__main__":
    from pathlib import Path
    
    # Create assets directory if needed
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    
    create_test_image()
