#!/usr/bin/env python3
import argparse
import logging
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline_r1b import Round1BPipeline

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(
        description='Adobe Hackathon Round 1B - Persona-Driven Document Intelligence'
    )
    parser.add_argument(
        '--input-dir', 
        default='/app/input',
        help='Input directory containing PDFs'
    )
    parser.add_argument(
        '--output-dir', 
        default='/app/output',
        help='Output directory for results'
    )
    parser.add_argument(
        '--persona-file',
        default='/app/output/persona.json',
        help='JSON file containing persona and job description'
    )
    parser.add_argument(
        '--models-dir',
        default='models',
        help='Directory containing ML models'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--cleanup-models',
        action='store_true',
        help='Delete all downloaded models and temporary files after run (with confirmation)'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        setup_logging(logging.DEBUG)
    
    logger.info("Starting Adobe Hackathon Round 1B - Persona-Driven Document Intelligence")
    logger.info(f"Input: {args.input_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Models: {args.models_dir}")
    logger.info(f"Persona: {args.persona_file}")
    
    # Handle cleanup-only mode
    if args.cleanup_models:
        confirm = input('Are you sure you want to delete all models and cache files? [y/N]: ')
        if confirm.lower() == 'y':
            import shutil
            import glob
            model_dirs = ['models', '/opt/models']
            for d in model_dirs:
                try:
                    shutil.rmtree(d, ignore_errors=True)
                    logger.info(f"Deleted model directory: {d}")
                except Exception as e:
                    logger.warning(f"Could not delete {d}: {e}")
            # Optionally, clean HuggingFace cache if present
            for cache_dir in glob.glob(str(Path.home()) + '/.cache/huggingface*'):
                try:
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    logger.info(f"Deleted cache: {cache_dir}")
                except Exception as e:
                    logger.warning(f"Could not delete cache {cache_dir}: {e}")
        else:
            logger.info('Cleanup cancelled by user.')
        return

    # Verify input directory exists
    if not Path(args.input_dir).exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check for PDFs
    pdf_files = list(Path(args.input_dir).glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {args.input_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(pdf_files)} PDF file(s)")
    
    try:
        logger.info("Running Round 1B: Persona-Driven Document Intelligence")
        pipeline = Round1BPipeline(models_dir=args.models_dir)
        success = pipeline.process_batch(args.input_dir, args.output_dir, args.persona_file)
        
        if success:
            logger.info("✅ Round 1B completed successfully")
        else:
            logger.error("❌ Round 1B completed with errors")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
        confirm = input('Are you sure you want to delete all models and cache files? [y/N]: ')
        if confirm.lower() == 'y':
            import shutil
            import glob
            model_dirs = ['/opt/models', 'adobe-hackathon-backend/models']
            for d in model_dirs:
                try:
                    shutil.rmtree(d, ignore_errors=True)
                    logger.info(f"Deleted model directory: {d}")
                except Exception as e:
                    logger.warning(f"Could not delete {d}: {e}")
            # Optionally, clean HuggingFace cache if present
            for cache_dir in glob.glob(str(Path.home()) + '/.cache/huggingface*'):
                try:
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    logger.info(f"Deleted cache: {cache_dir}")
                except Exception as e:
                    logger.warning(f"Could not delete cache {cache_dir}: {e}")
        else:
            logger.info('Cleanup cancelled by user.')

if __name__ == "__main__":
    main() 