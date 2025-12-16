#!/usr/bin/env python3
"""Generate test images for VL quality rubric."""

from PIL import Image, ImageDraw, ImageFont
import os

OUTPUT_DIR = "/mnt/raid0/llm/claude/test_images/vl_rubric"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_font(size=40):
    """Get a font, falling back to default if needed."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except:
        return ImageFont.load_default()

# T1-Q1: Simple OCR
def create_text_simple():
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    font = get_font(48)
    draw.text((20, 25), "Hello World 123", fill='black', font=font)
    img.save(f"{OUTPUT_DIR}/text_simple.png")
    print("Created: text_simple.png")

# T1-Q2: Basic shapes
def create_shapes_basic():
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    # 3 red circles
    for i, x in enumerate([50, 120, 190]):
        draw.ellipse([x, 50, x+40, 90], fill='red', outline='darkred')
    # 2 blue squares
    for i, x in enumerate([280, 340]):
        draw.rectangle([x, 50, x+40, 90], fill='blue', outline='darkblue')
    img.save(f"{OUTPUT_DIR}/shapes_basic.png")
    print("Created: shapes_basic.png")

# T1-Q3: Folder icon (simple representation)
def create_icon_folder():
    img = Image.new('RGB', (100, 100), color='white')
    draw = ImageDraw.Draw(img)
    # Folder shape
    draw.rectangle([10, 30, 90, 85], fill='#FFD700', outline='#DAA520')
    draw.rectangle([10, 20, 40, 35], fill='#FFD700', outline='#DAA520')
    img.save(f"{OUTPUT_DIR}/icon_folder.png")
    print("Created: icon_folder.png")

# T2-Q1: Bar chart
def create_chart_bar():
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    font = get_font(20)

    # Axis
    draw.line([(50, 250), (350, 250)], fill='black', width=2)
    draw.line([(50, 250), (50, 30)], fill='black', width=2)

    # Bars: A=10, B=25, C=15, D=20
    values = [('A', 10), ('B', 25), ('C', 15), ('D', 20)]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']

    for i, (label, val) in enumerate(values):
        x = 80 + i * 70
        height = val * 8
        draw.rectangle([x, 250-height, x+50, 250], fill=colors[i])
        draw.text((x+15, 255), label, fill='black', font=font)
        draw.text((x+10, 250-height-25), str(val), fill='black', font=font)

    # Y-axis labels
    for v in [0, 10, 20, 30]:
        y = 250 - v * 8
        draw.text((25, y-10), str(v), fill='black', font=font)

    img.save(f"{OUTPUT_DIR}/chart_bar.png")
    print("Created: chart_bar.png")

# T2-Q2: Simple invoice
def create_doc_invoice():
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    font = get_font(16)
    font_bold = get_font(20)

    draw.text((150, 10), "INVOICE", fill='black', font=font_bold)
    draw.text((20, 50), "Date: 2025-12-16", fill='black', font=font)
    draw.line([(20, 80), (380, 80)], fill='black')

    # Header
    draw.text((20, 90), "Item", fill='black', font=font)
    draw.text((150, 90), "Qty", fill='black', font=font)
    draw.text((220, 90), "Price", fill='black', font=font)
    draw.text((300, 90), "Amount", fill='black', font=font)
    draw.line([(20, 115), (380, 115)], fill='black')

    # Items
    items = [("Widget A", 2, 15.00), ("Widget B", 3, 25.00), ("Service", 1, 50.00)]
    y = 125
    for item, qty, price in items:
        amount = qty * price
        draw.text((20, y), item, fill='black', font=font)
        draw.text((150, y), str(qty), fill='black', font=font)
        draw.text((220, y), f"${price:.2f}", fill='black', font=font)
        draw.text((300, y), f"${amount:.2f}", fill='black', font=font)
        y += 30

    draw.line([(20, y), (380, y)], fill='black')
    total = sum(qty * price for _, qty, price in items)
    draw.text((220, y+10), "TOTAL:", fill='black', font=font_bold)
    draw.text((300, y+10), f"${total:.2f}", fill='black', font=font_bold)

    img.save(f"{OUTPUT_DIR}/doc_invoice.png")
    print("Created: doc_invoice.png")

# T2-Q3: Code with bug
def create_code_python():
    img = Image.new('RGB', (500, 250), color='#1e1e1e')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
    except:
        font = get_font(16)

    code_lines = [
        ("def ", "#569cd6"), ("calculate_average", "#dcdcaa"), ("(numbers):", "#d4d4d4"),
        ("    total = ", "#d4d4d4"), ("0", "#b5cea8"),
        ("    for ", "#c586c0"), ("num ", "#9cdcfe"), ("in ", "#c586c0"), ("numbers:", "#d4d4d4"),
        ("        total += num", "#d4d4d4"),
        ("    average = total / ", "#d4d4d4"), ("len(numbers)", "#dcdcaa"),
        ("    return ", "#c586c0"), ("total", "#9cdcfe"), ("  # BUG: should return average", "#6a9955"),
    ]

    y = 20
    for i, line in enumerate(code_lines):
        if isinstance(line, tuple) and len(line) == 2:
            draw.text((20, y), line[0], fill=line[1], font=font)
        else:
            x = 20
            for text, color in [line[j:j+2] for j in range(0, len(line), 2)]:
                draw.text((x, y), text, fill=color, font=font)
                x += len(text) * 10
        y += 25

    img.save(f"{OUTPUT_DIR}/code_python.png")
    print("Created: code_python.png")

# T3-Q1: Math equation
def create_math_equation():
    img = Image.new('RGB', (300, 100), color='white')
    draw = ImageDraw.Draw(img)
    font = get_font(36)
    draw.text((30, 30), "2x + 5 = 13", fill='black', font=font)
    # Add "solve for x" instruction
    font_small = get_font(20)
    draw.text((30, 75), "Solve for x", fill='gray', font=font_small)
    img.save(f"{OUTPUT_DIR}/math_equation.png")
    print("Created: math_equation.png")

# T3-Q2: Flowchart
def create_diagram_flowchart():
    img = Image.new('RGB', (500, 400), color='white')
    draw = ImageDraw.Draw(img)
    font = get_font(14)

    # Start
    draw.ellipse([200, 10, 300, 50], outline='black', width=2)
    draw.text((225, 20), "START", fill='black', font=font)

    # Diamond: input > 10?
    draw.polygon([(250, 80), (320, 130), (250, 180), (180, 130)], outline='black', width=2)
    draw.text((210, 115), "input > 10?", fill='black', font=font)

    # Yes branch - Diamond: flag = true?
    draw.polygon([(380, 130), (450, 180), (380, 230), (310, 180)], outline='black', width=2)
    draw.text((335, 165), "flag=true?", fill='black', font=font)

    # No branch from first diamond
    draw.rectangle([50, 110, 150, 150], outline='black', width=2)
    draw.text((65, 120), "Path A", fill='black', font=font)

    # Yes from second diamond
    draw.rectangle([350, 270, 450, 310], outline='black', width=2)
    draw.text((365, 280), "Path B", fill='black', font=font)

    # No from second diamond
    draw.rectangle([200, 270, 300, 310], outline='black', width=2)
    draw.text((215, 280), "Path C", fill='black', font=font)

    # End
    draw.ellipse([200, 340, 300, 380], outline='black', width=2)
    draw.text((230, 350), "END", fill='black', font=font)

    # Arrows
    draw.line([(250, 50), (250, 80)], fill='black', width=2)
    draw.line([(320, 130), (380, 130)], fill='black', width=2)  # Yes to second diamond
    draw.text((335, 110), "Yes", fill='green', font=font)
    draw.line([(180, 130), (150, 130)], fill='black', width=2)  # No to Path A
    draw.text((155, 110), "No", fill='red', font=font)
    draw.line([(380, 230), (380, 270)], fill='black', width=2)  # Yes to Path B
    draw.text((385, 240), "Yes", fill='green', font=font)
    draw.line([(310, 180), (250, 180), (250, 270)], fill='black', width=2)  # No to Path C
    draw.text((260, 220), "No", fill='red', font=font)
    draw.line([(250, 310), (250, 340)], fill='black', width=2)
    draw.line([(400, 310), (400, 360), (300, 360)], fill='black', width=2)
    draw.line([(100, 150), (100, 360), (200, 360)], fill='black', width=2)

    img.save(f"{OUTPUT_DIR}/diagram_flowchart.png")
    print("Created: diagram_flowchart.png")

# T3-Q3: Spot the difference
def create_diff_images():
    # Create two images side by side with 3 differences
    img = Image.new('RGB', (500, 200), color='white')
    draw = ImageDraw.Draw(img)
    font = get_font(16)

    # Left image
    draw.rectangle([20, 20, 220, 180], outline='black')
    draw.text((90, 5), "Image A", fill='black', font=font)
    draw.ellipse([50, 50, 100, 100], fill='red')  # Red circle
    draw.rectangle([120, 50, 170, 100], fill='blue')  # Blue square
    draw.ellipse([80, 120, 130, 160], fill='green')  # Green circle

    # Right image (3 differences)
    draw.rectangle([280, 20, 480, 180], outline='black')
    draw.text((350, 5), "Image B", fill='black', font=font)
    draw.ellipse([310, 50, 360, 100], fill='yellow')  # Diff 1: yellow instead of red
    draw.rectangle([380, 50, 430, 100], fill='blue')  # Same blue square
    draw.rectangle([340, 120, 390, 160], fill='green')  # Diff 2: square instead of circle
    # Diff 3: missing element (no third shape in same position)
    draw.ellipse([400, 130, 440, 170], fill='purple')  # Extra purple circle

    img.save(f"{OUTPUT_DIR}/diff_images.png")
    print("Created: diff_images.png")

# T3-Q4: Pattern puzzle
def create_puzzle_grid():
    img = Image.new('RGB', (300, 300), color='white')
    draw = ImageDraw.Draw(img)
    font = get_font(20)

    # 3x3 grid
    cell_size = 80
    offset = 30

    for i in range(4):
        draw.line([(offset, offset + i*cell_size), (offset + 3*cell_size, offset + i*cell_size)], fill='black', width=2)
        draw.line([(offset + i*cell_size, offset), (offset + i*cell_size, offset + 3*cell_size)], fill='black', width=2)

    # Pattern: Row 1: circle, square, triangle
    #          Row 2: square, triangle, circle
    #          Row 3: triangle, circle, ?
    shapes = [
        ['circle', 'square', 'triangle'],
        ['square', 'triangle', 'circle'],
        ['triangle', 'circle', '?']
    ]

    for row in range(3):
        for col in range(3):
            x = offset + col * cell_size + cell_size // 2
            y = offset + row * cell_size + cell_size // 2
            shape = shapes[row][col]

            if shape == 'circle':
                draw.ellipse([x-25, y-25, x+25, y+25], outline='blue', width=3)
            elif shape == 'square':
                draw.rectangle([x-25, y-25, x+25, y+25], outline='red', width=3)
            elif shape == 'triangle':
                draw.polygon([(x, y-25), (x-25, y+25), (x+25, y+25)], outline='green', width=3)
            elif shape == '?':
                draw.text((x-10, y-15), "?", fill='gray', font=get_font(40))

    img.save(f"{OUTPUT_DIR}/puzzle_grid.png")
    print("Created: puzzle_grid.png")

if __name__ == "__main__":
    print(f"Generating test images in {OUTPUT_DIR}/\n")

    # T1
    create_text_simple()
    create_shapes_basic()
    create_icon_folder()

    # T2
    create_chart_bar()
    create_doc_invoice()
    create_code_python()

    # T3
    create_math_equation()
    create_diagram_flowchart()
    create_diff_images()
    create_puzzle_grid()

    print(f"\nDone! {len(os.listdir(OUTPUT_DIR))} images created.")
