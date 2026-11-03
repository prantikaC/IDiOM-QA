import os
import sys
import io
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.graph import build_sequential_graph

def plot_workflow():
    print(f"{'='*50}")
    print("GENERATING LANGGRAPH WORKFLOW PLOT (PDF)")
    print(f"{'='*50}\n")

    # 1. Compile the graph
    app = build_sequential_graph()
    
    # 2. Define where to save the document (saving it in the root folder)
    output_path = os.path.join(os.getcwd(), "workflow_architecture.pdf")
    
    try:
        # 3. Generate the PNG byte data using LangGraph's built-in Mermaid generator
        print("--- Contacting Mermaid API to render graph ---")
        png_data = app.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)
        
        # 4. Load the bytes into memory, convert to RGB, and save as PDF
        image = Image.open(io.BytesIO(png_data))
        
        # Strip the alpha channel (transparency) as PDFs do not support it natively via Pillow
        rgb_image = image.convert("RGB")
        
        rgb_image.save(output_path, "PDF", resolution=100.0)
            
        print(f"\nSuccess! Workflow successfully plotted and saved to:")
        print(f"{output_path}")
        
    except Exception as e:
        print(f"\nFailed to generate PDF: {e}")
        print("\n--- Fallback: Raw Mermaid Code ---")
        print("You can copy and paste the code below into https://mermaid.live/ to view your graph:\n")
        print(app.get_graph().draw_mermaid())

if __name__ == "__main__":
    plot_workflow()