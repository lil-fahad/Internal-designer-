"""
Example: Generate interior design from a floor plan
"""
import sys
import os
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import InteriorDesigner
from config import Config


def create_sample_floor_plan():
    """Create a sample floor plan for demonstration"""
    # Create a simple floor plan (white background with black walls)
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    
    # Draw outer walls (black)
    img[0:10, :] = 0  # Top wall
    img[-10:, :] = 0  # Bottom wall
    img[:, 0:10] = 0  # Left wall
    img[:, -10:] = 0  # Right wall
    
    # Draw interior walls
    img[250:260, :] = 0  # Horizontal wall
    img[:, 250:260] = 0  # Vertical wall
    
    return img


def main():
    """Generate interior design"""
    print("="*60)
    print("AI Interior Designer - Design Generation")
    print("="*60)
    
    # Initialize designer
    print("\nInitializing AI Interior Designer...")
    designer = InteriorDesigner()
    
    # Create or load floor plan
    print("\nCreating sample floor plan...")
    floor_plan = create_sample_floor_plan()
    
    # Save floor plan for reference
    floor_plan_img = Image.fromarray(floor_plan)
    floor_plan_img.save('sample_floor_plan.png')
    print("Sample floor plan saved to: sample_floor_plan.png")
    
    # Analyze floor plan
    print("\nAnalyzing floor plan...")
    analysis = designer.analyze_floor_plan(floor_plan)
    print(f"  Number of rooms detected: {analysis['num_rooms']}")
    print(f"  Total area: {analysis['total_area']}")
    print(f"  Floor plan dimensions: {analysis['dimensions']}")
    
    # Generate designs
    print("\nGenerating interior designs...")
    styles = ['modern', 'classic', 'minimalist']
    
    for style in styles:
        print(f"\n  Generating {style} design...")
        designs = designer.generate_design(
            floor_plan, 
            style=style, 
            num_variations=2
        )
        
        # Save generated designs
        for idx, design in enumerate(designs):
            design_img = Image.fromarray(design)
            filename = f'design_{style}_v{idx+1}.png'
            design_img.save(filename)
            print(f"    Saved: {filename}")
    
    # Get furniture suggestions
    print("\n\nGetting furniture suggestions...")
    room_types = ['bedroom', 'living_room', 'kitchen']
    
    for room_type in room_types:
        print(f"\n  {room_type.replace('_', ' ').title()}:")
        suggestions = designer.suggest_furniture(
            room_type=room_type,
            room_dimensions=(5, 3, 4),
            style='modern'
        )
        
        for item in suggestions[:3]:  # Show first 3 items
            print(f"    - {item['item']} ({item.get('size', 'standard')})")
    
    print("\n" + "="*60)
    print("✓ Design generation completed successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - sample_floor_plan.png")
    for style in styles:
        print(f"  - design_{style}_v1.png")
        print(f"  - design_{style}_v2.png")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
