"""
Generates rich, high-resolution aesthetic sample images using PIL for the multimodal vector database.
"""
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def create_gradient_canvas(width, height, top_color, bottom_color):
    """Creates a vertical linear gradient image."""
    base = Image.new('RGB', (width, height), top_color)
    top = np.array(top_color, dtype=np.float32)
    bottom = np.array(bottom_color, dtype=np.float32)
    
    # Generate vertical gradient
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        factor = y / float(height - 1)
        c = (1.0 - factor) * top + factor * bottom
        arr[y, :, :] = c.astype(np.uint8)
    return Image.fromarray(arr)

def generate_sunset_beach(path):
    w, h = 600, 450
    # Warm golden orange to deep sunset purple
    img = create_gradient_canvas(w, h, (40, 20, 60), (255, 120, 40))
    draw = ImageDraw.Draw(img)
    
    # Golden sun glow
    sun_x, sun_y = 300, 240
    for r in range(90, 10, -5):
        alpha = int(255 * (1 - r/90))
        draw.ellipse([sun_x - r, sun_y - r, sun_x + r, sun_y + r], fill=(255, 230, 150))
    draw.ellipse([sun_x - 30, sun_y - 30, sun_x + 30, sun_y + 30], fill=(255, 250, 220))
    
    # Ocean water from horizon down
    ocean = create_gradient_canvas(w, h - 250, (220, 90, 40), (20, 40, 80))
    img.paste(ocean, (0, 250))
    
    draw = ImageDraw.Draw(img)
    # Sun reflection ripples on ocean
    for y in range(255, h - 40, 8):
        rw = (y - 250) * 1.5 + 20
        draw.line([(sun_x - rw, y), (sun_x + rw, y)], fill=(255, 210, 130), width=2)
        
    # Dark beach sand bottom
    draw.polygon([(0, h), (0, h - 70), (w, h - 40), (w, h)], fill=(30, 20, 25))
    
    # Palm silhouette
    draw.line([(520, h - 50), (480, 200), (450, 120)], fill=(15, 10, 15), width=10)
    for angle in [20, 60, 100, 140, 180, 220]:
        rad = math.radians(angle)
        dx = int(math.cos(rad) * 120)
        dy = int(math.sin(rad) * 60)
        draw.arc([450 + dx - 60, 120 + dy - 30, 450 + dx + 60, 120 + dy + 30], 0, 180, fill=(15, 10, 15), width=5)
    
    img.save(path, quality=95)

def generate_mountain_snow(path):
    w, h = 600, 450
    # Clean alpine sky
    img = create_gradient_canvas(w, h, (120, 175, 230), (210, 235, 255))
    draw = ImageDraw.Draw(img)
    
    # Distant peaks
    draw.polygon([(50, h), (200, 160), (320, 280), (450, 140), (600, h)], fill=(180, 205, 230))
    # Mid mountain range
    draw.polygon([(0, h), (140, 120), (260, 260), (390, 90), (520, 240), (600, 180), (600, h)], fill=(140, 160, 190))
    
    # Snow caps (bright crisp white)
    draw.polygon([(140, 120), (110, 170), (140, 160), (160, 180)], fill=(255, 255, 255))
    draw.polygon([(390, 90), (340, 170), (380, 150), (430, 180)], fill=(255, 255, 255))
    draw.polygon([(450, 140), (420, 190), (470, 180)], fill=(245, 250, 255))
    
    # Foreground snowy slope
    draw.polygon([(0, h), (0, 320), (300, 280), (600, 350), (600, h)], fill=(240, 248, 255))
    # Rocky shadows
    draw.polygon([(180, 310), (220, 340), (260, 300)], fill=(100, 120, 145))
    draw.polygon([(420, 360), (480, 380), (450, 350)], fill=(110, 125, 150))
    
    img.save(path, quality=95)

def generate_forest_mist(path):
    w, h = 600, 450
    # Misty emerald green gradient
    img = create_gradient_canvas(w, h, (200, 225, 215), (70, 110, 90))
    draw = ImageDraw.Draw(img)
    
    # Layers of pine tree silhouettes
    # Back layer (pale sage)
    for x in range(20, w + 40, 45):
        top_y = 130 + (x % 30)
        draw.polygon([(x, top_y), (x - 30, h), (x + 30, h)], fill=(130, 165, 150))
        
    # Mist band
    mist = Image.new('RGBA', (w, 100), (220, 235, 230, 120))
    mist = mist.filter(ImageFilter.GaussianBlur(10))
    img.paste(mist, (0, 220), mist)
    
    # Middle layer (darker evergreen)
    draw = ImageDraw.Draw(img)
    for x in range(10, w + 60, 60):
        top_y = 170 + ((x * 7) % 40)
        draw.polygon([(x, top_y), (x - 40, h), (x + 40, h)], fill=(60, 95, 75))
        
    # Front layer (deep pine green)
    for x in range(-20, w + 80, 80):
        top_y = 210 + ((x * 13) % 50)
        draw.polygon([(x, top_y), (x - 55, h), (x + 55, h)], fill=(25, 55, 40))
        
    img.save(path, quality=95)

def generate_modern_skyscraper(path):
    w, h = 600, 450
    # Bright sky
    img = create_gradient_canvas(w, h, (90, 150, 220), (220, 235, 250))
    draw = ImageDraw.Draw(img)
    
    # Main glass skyscraper (perspective looking up)
    # Left facade (reflective steel blue)
    draw.polygon([(180, 40), (320, 20), (320, h), (180, h)], fill=(160, 195, 230))
    # Right facade (shadowed architectural glass)
    draw.polygon([(320, 20), (460, 70), (460, h), (320, h)], fill=(90, 130, 175))
    
    # Glass panel window grid
    for y in range(40, h, 22):
        # Left grid
        draw.line([(180, y), (320, y - 5)], fill=(225, 240, 255), width=1)
        # Right grid
        draw.line([(320, y - 5), (460, y + 10)], fill=(60, 95, 135), width=1)
        
    # Vertical mullions
    for fx in [0.25, 0.5, 0.75]:
        lx = int(180 + fx * (320 - 180))
        draw.line([(lx, 35), (lx, h)], fill=(210, 230, 250), width=1)
        rx = int(320 + fx * (460 - 320))
        draw.line([(rx, 35), (rx, h)], fill=(75, 110, 150), width=1)
        
    # Background neighboring tower
    draw.polygon([(80, 150), (160, 130), (160, h), (80, h)], fill=(185, 205, 225))
    draw.polygon([(470, 180), (550, 200), (550, h), (470, h)], fill=(140, 170, 200))
    
    img.save(path, quality=95)

def generate_ancient_temple(path):
    w, h = 600, 450
    # Warm mediterranean sky
    img = create_gradient_canvas(w, h, (140, 185, 225), (245, 230, 205))
    draw = ImageDraw.Draw(img)
    
    # Temple base steps
    draw.rectangle([100, 360, 500, 380], fill=(215, 190, 155))
    draw.rectangle([80, 380, 520, 400], fill=(195, 170, 135))
    draw.rectangle([60, 400, 540, 425], fill=(175, 150, 120))
    
    # Pediment / Roof Triangle
    draw.polygon([(110, 170), (300, 80), (490, 170)], fill=(225, 200, 165))
    draw.polygon([(130, 165), (300, 95), (470, 165)], fill=(180, 155, 125))
    draw.rectangle([105, 170, 495, 195], fill=(210, 185, 150))
    
    # Classical stone columns
    for cx in [140, 200, 260, 340, 400, 460]:
        # Column shadow and highlight
        draw.rectangle([cx - 14, 195, cx + 14, 360], fill=(230, 205, 170))
        draw.rectangle([cx - 14, 195, cx - 7, 360], fill=(200, 175, 140))
        # Column cap
        draw.rectangle([cx - 18, 190, cx + 18, 200], fill=(215, 190, 155))
        draw.rectangle([cx - 18, 355, cx + 18, 365], fill=(215, 190, 155))
        
    img.save(path, quality=95)

def generate_cozy_coffee(path):
    w, h = 600, 450
    # Warm dark walnut cafe background
    img = create_gradient_canvas(w, h, (45, 30, 25), (75, 50, 35))
    draw = ImageDraw.Draw(img)
    
    # Wooden table surface
    draw.rectangle([0, 250, w, h], fill=(120, 75, 45))
    for y in range(250, h, 20):
        draw.line([(0, y), (w, y)], fill=(95, 55, 30), width=2)
        
    # Ceramic coffee cup saucer
    draw.ellipse([200, 280, 400, 360], fill=(235, 230, 220))
    draw.ellipse([210, 290, 390, 350], fill=(210, 205, 195))
    
    # White ceramic cup body
    draw.ellipse([230, 230, 370, 320], fill=(245, 245, 240))
    # Cup handle
    draw.arc([350, 245, 400, 300], -60, 90, fill=(235, 235, 230), width=10)
    
    # Rich espresso crema with heart latte art
    draw.ellipse([240, 240, 360, 310], fill=(130, 75, 35))
    draw.ellipse([250, 250, 350, 300], fill=(100, 55, 25))
    # Heart latte art in white cream
    draw.ellipse([275, 260, 305, 285], fill=(245, 235, 215))
    draw.ellipse([295, 260, 325, 285], fill=(245, 235, 215))
    draw.polygon([(275, 275), (325, 275), (300, 295)], fill=(245, 235, 215))
    
    # Roasted coffee beans on table
    for bx, by in [(130, 340), (150, 355), (170, 335), (430, 345), (455, 330), (470, 360)]:
        draw.ellipse([bx, by, bx + 18, by + 12], fill=(45, 25, 15))
        draw.line([(bx + 3, by + 6), (bx + 15, by + 6)], fill=(20, 10, 5), width=1)
        
    img.save(path, quality=95)

def generate_neon_night_city(path):
    w, h = 600, 450
    # Deep midnight cyberpunk city
    img = create_gradient_canvas(w, h, (10, 10, 25), (25, 15, 40))
    draw = ImageDraw.Draw(img)
    
    # City silhouette
    for bx, bw, bh in [(30, 80, 250), (130, 60, 320), (210, 90, 280), (330, 70, 340), (420, 100, 290), (530, 60, 220)]:
        draw.rectangle([bx, h - bh, bx + bw, h], fill=(20, 20, 35))
        # Windows
        for wy in range(h - bh + 20, h - 80, 18):
            for wx in range(bx + 10, bx + bw - 10, 14):
                if (wx + wy) % 5 != 0:
                    c = (255, 220, 120) if (wx % 3 == 0) else (100, 200, 255)
                    draw.rectangle([wx, wy, wx + 6, wy + 8], fill=c)
                    
    # Wet street asphalt reflecting neon
    draw.rectangle([0, h - 80, w, h], fill=(15, 12, 22))
    
    # Neon signs (Electric Cyan & Hot Magenta)
    # Cyan billboard
    draw.rectangle([140, 180, 180, 240], fill=(0, 220, 255))
    draw.line([(140, 210), (180, 210)], fill=(255, 255, 255), width=2)
    # Magenta glowing sign
    draw.rectangle([340, 150, 390, 210], fill=(255, 20, 140))
    draw.line([(340, 180), (390, 180)], fill=(255, 255, 255), width=2)
    
    # Wet road reflections
    for ry in range(h - 75, h, 6):
        draw.line([(140 - 20, ry), (180 + 20, ry)], fill=(0, 180, 230), width=2)
        draw.line([(340 - 25, ry), (390 + 25, ry)], fill=(230, 20, 130), width=2)
        
    img.save(path, quality=95)

def generate_desert_dunes(path):
    w, h = 600, 450
    # Clear warm azure desert sky
    img = create_gradient_canvas(w, h, (100, 175, 240), (250, 230, 190))
    draw = ImageDraw.Draw(img)
    
    # Distant dune ridge
    draw.polygon([(0, h), (0, 260), (240, 210), (450, 250), (600, 200), (600, h)], fill=(235, 180, 100))
    # Mid-range dramatic crest
    draw.polygon([(0, h), (0, 290), (320, 230), (600, 310), (600, h)], fill=(245, 195, 115))
    # Crest shadow side
    draw.polygon([(320, 230), (450, 340), (600, 310)], fill=(205, 145, 75))
    
    # Foreground sweeping sand dune
    draw.polygon([(0, h), (0, 340), (220, 290), (480, 390), (600, 360), (600, h)], fill=(250, 205, 130))
    draw.polygon([(220, 290), (350, 420), (480, 390)], fill=(195, 130, 60))
    
    # Wind ripple lines
    for y in range(350, h, 14):
        draw.arc([100, y, 300, y + 25], 0, 180, fill=(235, 180, 105), width=1)
        
    img.save(path, quality=95)

def generate_vintage_camera(path):
    w, h = 600, 450
    # Studio soft neutral backdrop
    img = create_gradient_canvas(w, h, (240, 240, 245), (200, 205, 215))
    draw = ImageDraw.Draw(img)
    
    # Camera soft drop shadow
    draw.ellipse([140, 340, 460, 390], fill=(160, 165, 175))
    
    # Camera body: Chrome top plate
    draw.rectangle([160, 140, 440, 180], fill=(225, 228, 232))
    draw.rectangle([160, 178, 440, 182], fill=(180, 185, 190))
    
    # Viewfinder window & dials
    draw.rectangle([180, 150, 210, 170], fill=(40, 45, 55))
    draw.rectangle([400, 125, 425, 140], fill=(210, 215, 220)) # Shutter button
    draw.rectangle([230, 130, 265, 140], fill=(190, 195, 200)) # Dial
    
    # Black textured leatherette body
    draw.rectangle([160, 182, 440, 320], fill=(35, 38, 42))
    
    # Circular 50mm Lens mount & glass
    cx, cy = 300, 245
    # Outer chrome ring
    draw.ellipse([cx - 75, cy - 75, cx + 75, cy + 75], fill=(215, 220, 225))
    draw.ellipse([cx - 70, cy - 70, cx + 70, cy + 70], fill=(60, 65, 70))
    # Dark glass elements with anti-reflective coating reflection
    draw.ellipse([cx - 55, cy - 55, cx + 55, cy + 55], fill=(25, 30, 40))
    draw.ellipse([cx - 40, cy - 40, cx + 40, cy + 40], fill=(15, 20, 30))
    # Cyan/magenta optic lens reflection
    draw.arc([cx - 35, cy - 35, cx + 35, cy + 35], -40, 40, fill=(80, 180, 230), width=4)
    draw.arc([cx - 25, cy - 25, cx + 25, cy + 25], 140, 220, fill=(220, 80, 180), width=3)
    
    img.save(path, quality=95)

def generate_coral_reef(path):
    w, h = 600, 450
    # Turquoise tropical ocean gradient
    img = create_gradient_canvas(w, h, (40, 180, 220), (10, 60, 120))
    draw = ImageDraw.Draw(img)
    
    # Ocean floor
    draw.rectangle([0, 380, w, h], fill=(220, 205, 170))
    
    # Coral structures: vibrant pink, magenta, orange
    for cx, cy, col in [(100, 380, (255, 105, 140)), (240, 390, (255, 140, 50)), (380, 370, (230, 80, 170)), (500, 385, (255, 180, 70))]:
        # Branching coral fans
        for angle in [-60, -30, 0, 30, 60]:
            rad = math.radians(angle)
            ex = int(cx + math.sin(rad) * 90)
            ey = int(cy - math.cos(rad) * 90)
            draw.line([(cx, cy), (ex, ey)], fill=col, width=12)
            draw.ellipse([ex - 10, ey - 10, ex + 10, ey + 10], fill=col)
            
    # Tropical yellow fish
    for fx, fy in [(180, 160), (320, 210), (440, 140)]:
        draw.polygon([(fx, fy), (fx - 30, fy - 15), (fx - 30, fy + 15)], fill=(255, 215, 0))
        draw.polygon([(fx - 30, fy), (fx - 45, fy - 12), (fx - 45, fy + 12)], fill=(255, 140, 0))
        draw.ellipse([fx - 8, fy - 4, fx - 4, fy], fill=(0, 0, 0))
        
    img.save(path, quality=95)

def generate_autumn_park(path):
    w, h = 600, 450
    # Soft autumn sky
    img = create_gradient_canvas(w, h, (190, 215, 235), (240, 230, 215))
    draw = ImageDraw.Draw(img)
    
    # Ground & Winding Stone Path
    draw.rectangle([0, 270, w, h], fill=(110, 120, 70)) # Green/amber grass
    draw.polygon([(180, h), (420, h), (320, 270), (280, 270)], fill=(190, 180, 165)) # Pathway
    
    # Maple trees with red, orange, and golden crowns
    trees = [
        (80, 220, (215, 50, 35)),   # Crimson red
        (180, 200, (240, 120, 30)), # Vivid orange
        (420, 200, (245, 180, 25)), # Golden yellow
        (520, 220, (190, 45, 25))   # Dark amber
    ]
    for tx, ty, col in trees:
        # Trunk
        draw.rectangle([tx - 8, ty, tx + 8, ty + 80], fill=(70, 50, 40))
        # Leaf foliage cluster
        draw.ellipse([tx - 65, ty - 80, tx + 65, ty + 20], fill=col)
        draw.ellipse([tx - 40, ty - 100, tx + 40, ty - 20], fill=(min(col[0] + 20, 255), col[1], col[2]))
        
    # Scattered leaves on path
    for lx, ly, c in [(240, 360, (215, 50, 35)), (340, 390, (240, 120, 30)), (290, 330, (245, 180, 25)), (360, 310, (215, 50, 35))]:
        draw.ellipse([lx, ly, lx + 10, ly + 6], fill=c)
        
    img.save(path, quality=95)

def generate_minimal_interior(path):
    w, h = 600, 450
    # Clean bright white Scandinavian interior
    img = Image.new('RGB', (w, h), (248, 249, 250))
    draw = ImageDraw.Draw(img)
    
    # Wall corner and natural oak wood floor
    draw.line([(220, 0), (220, 310)], fill=(225, 228, 232), width=1)
    draw.polygon([(0, 310), (w, 310), (w, h), (0, h)], fill=(235, 220, 200)) # Oak floor
    for y in range(310, h, 24):
        draw.line([(0, y), (w, y)], fill=(215, 200, 180), width=1)
        
    # Sunlight shadow cast diagonally across the wall and floor
    sun_poly = [(350, 0), (550, 0), (450, h), (200, h)]
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.polygon(sun_poly, fill=(255, 255, 240, 60))
    img.paste(Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB'))
    
    draw = ImageDraw.Draw(img)
    # Designer wood & black minimalist chair
    # Legs
    draw.line([(290, 280), (275, 390)], fill=(40, 40, 40), width=4)
    draw.line([(350, 280), (365, 390)], fill=(40, 40, 40), width=4)
    draw.line([(310, 260), (300, 370)], fill=(50, 50, 50), width=3)
    draw.line([(340, 260), (350, 370)], fill=(50, 50, 50), width=3)
    # Seat
    draw.rectangle([280, 275, 360, 288], fill=(190, 150, 110))
    # Curved backrest
    draw.rectangle([285, 220, 355, 235], fill=(190, 150, 110))
    draw.line([(290, 235), (290, 275)], fill=(40, 40, 40), width=3)
    draw.line([(350, 235), (350, 275)], fill=(40, 40, 40), width=3)
    
    # Small ceramic pot with minimalist ficus / monstera leaf
    draw.rectangle([130, 285, 170, 330], fill=(240, 240, 240))
    draw.line([(150, 285), (150, 220)], fill=(50, 120, 70), width=3)
    draw.ellipse([120, 200, 180, 235], fill=(60, 140, 80))
    
    img.save(path, quality=95)

def generate_all_catalog_images(output_dir="static/images"):
    os.makedirs(output_dir, exist_ok=True)
    catalog = [
        ("sunset_beach.jpg", generate_sunset_beach),
        ("mountain_snow.jpg", generate_mountain_snow),
        ("forest_mist.jpg", generate_forest_mist),
        ("modern_skyscraper.jpg", generate_modern_skyscraper),
        ("ancient_temple.jpg", generate_ancient_temple),
        ("cozy_coffee.jpg", generate_cozy_coffee),
        ("neon_night_city.jpg", generate_neon_night_city),
        ("desert_dunes.jpg", generate_desert_dunes),
        ("vintage_camera.jpg", generate_vintage_camera),
        ("coral_reef.jpg", generate_coral_reef),
        ("autumn_park.jpg", generate_autumn_park),
        ("minimal_interior.jpg", generate_minimal_interior),
    ]
    
    print(f"Generating {len(catalog)} rich catalog images...")
    for filename, func in catalog:
        filepath = os.path.join(output_dir, filename)
        func(filepath)
        print(f" -> Created {filepath}")
    print("Catalog image generation complete.")

if __name__ == "__main__":
    generate_all_catalog_images()
