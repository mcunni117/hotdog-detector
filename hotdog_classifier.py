import os
# Set this at the very top of the file, before any other imports
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from fastai.vision.all import *
from pathlib import Path
from duckduckgo_search import DDGS
from fastdownload import download_url
from loguru import logger
import sys
import os
import time

# Configure logging
logger.remove()
logger.add(
    "hotdog_classifier.log",
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    level="DEBUG",
    rotation="10 MB"
)
logger.add(sys.stderr, level="INFO")

def search_images(term, max_images=30):
    """Search for images using DuckDuckGo"""
    logger.info(f"Searching for '{term}'...")
    try:
        ddgs = DDGS()
        results = list(ddgs.images(
            term,
            max_results=max_images
        ))
        
        if not results:
            logger.warning(f"No results found for term: {term}")
            return L([])
            
        urls = L([r['image'] for r in results if r.get('image')])
        logger.success(f"Found {len(urls)} images for '{term}'")
        return urls
    except Exception as e:
        logger.error(f"Failed to search for '{term}': {str(e)}")
        raise

def download_images(urls, dest, max_pics=30, start_index=0):
    """Download images from urls to destination folder"""
    logger.info(f"Downloading {min(len(urls), max_pics)} images to {dest}")
    success_count = 0
    retry_count = 3  # Number of retries per URL
    
    # Import required for SSL handling
    import ssl
    import urllib.request
    ssl._create_default_https_context = ssl._create_unverified_context
    
    for i, url in enumerate(urls[:max_pics]):
        if i >= max_pics: break
        
        # Try multiple times for each URL
        for attempt in range(retry_count):
            try:
                # Use start_index + success_count for unique filenames
                fname = f'{start_index + success_count:03d}.jpg'
                dest_path = dest/fname
                
                logger.debug(f"Attempting to download {url} (attempt {attempt + 1}/{retry_count})")
                download_url(url, dest_path, show_progress=False)
                
                # Verify the image is valid
                try:
                    img = PILImage.create(dest_path)
                    # Check if image is too small
                    if min(img.size) < 100:
                        raise ValueError("Image too small")
                    success_count += 1
                    logger.debug(f"Successfully downloaded and verified image {success_count}/{max_pics}")
                    time.sleep(0.5)
                    break  # Success, move to next URL
                except Exception as e:
                    logger.warning(f"Downloaded file is not a valid image, removing: {e}")
                    os.remove(dest_path)
                    continue
                    
            except Exception as e:
                logger.warning(f'Failed download attempt {attempt + 1} for {url}: {str(e)}')
                if attempt == retry_count - 1:  # Last attempt
                    logger.warning(f'Skipping URL after {retry_count} failed attempts')
        
        # Log progress every 5 successful downloads
        if success_count > 0 and success_count % 5 == 0:
            logger.info(f"Progress: {success_count}/{max_pics} images successfully downloaded")
    
    if success_count == 0:
        logger.error("Failed to download any valid images!")
        raise Exception("No images were downloaded successfully")
        
    logger.success(f"Successfully downloaded {success_count} valid images")
    return success_count

def create_hotdog_classifier():
    logger.info("Starting hotdog classifier creation")
    path = Path('hotdog_data')
    
    try:
        # Always recreate not_hotdog if it has fewer than required images
        if not path.exists() or len(list((path/'not_hotdog').glob('*.jpg'))) < 20:
            logger.info("Creating/updating dataset directories")
            path.mkdir(exist_ok=True)
            (path/'hotdog').mkdir(exist_ok=True)
            (path/'not_hotdog').mkdir(exist_ok=True)
            
            searches = {
                'hotdog': ['hotdog food', 'hot dog food photo', 'hotdog sandwich'],
                'not_hotdog': [
                    'hamburger food',  # Changed from 'burger food'
                    'sandwich food close up',  # Added 'close up'
                    'taco food plate',  # Added 'plate'
                    'pizza slice',  # Changed from 'pizza food'
                    'chicken sandwich',  # New term
                    'french fries plate'  # New term
                ]
            }
            
            min_images_required = 20
            max_per_term = 10  # Maximum images per search term
            
            for label, terms in searches.items():
                if label == 'not_hotdog' or len(list((path/label).glob('*.jpg'))) < min_images_required:
                    logger.info(f"Processing {label} category")
                    dest = path/label
                    total_downloaded = 0
                    file_index = 0  # Add this to track overall file index
                    
                    for term in terms:
                        if total_downloaded >= min_images_required:
                            break
                            
                        try:
                            logger.info(f"Searching for term: {term}")
                            urls = search_images(term, max_images=20)
                            
                            # Pass the current file_index to download_images
                            downloaded = download_images(urls, dest, max_pics=max_per_term, start_index=file_index)
                            total_downloaded += downloaded
                            file_index += downloaded  # Increment file_index by number of downloaded images
                            
                            logger.info(f"Total {label} images so far: {total_downloaded}")
                            
                        except Exception as e:
                            logger.warning(f"Error processing term '{term}': {str(e)}")
                            continue
                    
                    if total_downloaded < min_images_required:
                        raise Exception(f"Failed to download enough images for {label} (got {total_downloaded}, need {min_images_required})")
        
        # Verify we have enough images and they're valid
        hotdog_images = list((path/'hotdog').glob('*.jpg'))
        not_hotdog_images = list((path/'not_hotdog').glob('*.jpg'))
        
        logger.info(f"Found {len(hotdog_images)} hotdog images and {len(not_hotdog_images)} not-hotdog images")
        
        if len(hotdog_images) < 10 or len(not_hotdog_images) < 10:
            raise Exception(f"Not enough images found in dataset (hotdog: {len(hotdog_images)}, not_hotdog: {len(not_hotdog_images)})")
        
        # Verify images are valid
        logger.info("Verifying images...")
        for img_path in hotdog_images + not_hotdog_images:
            try:
                PILImage.create(img_path)
            except Exception as e:
                logger.warning(f"Removing invalid image {img_path}: {str(e)}")
                os.remove(img_path)
        
        # Create and configure DataBlock with error handling
        logger.info("Creating DataBlock")
        hotdog_data = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            splitter=RandomSplitter(valid_pct=0.2, seed=42),
            get_y=parent_label,
            item_tfms=Resize(224),  # Simplified transform
            batch_tfms=aug_transforms(size=224, min_scale=0.75)
        )
        
        # Test the DataBlock before creating DataLoaders
        logger.info("Testing DataBlock")
        hotdog_data.summary(path)
        
        # Create DataLoaders with smaller batch size
        logger.info("Creating DataLoaders")
        dls = hotdog_data.dataloaders(
            path,
            batch_size=4,  # Smaller batch size
            num_workers=0  # No multiprocessing for debugging
        )
        
        logger.info(f"Dataset size - Training: {len(dls.train_ds)}, Validation: {len(dls.valid_ds)}")
        
        # Show a batch to verify data
        logger.info("Verifying batch creation")
        dls.show_batch(max_n=4)
        
        # Create and train model
        logger.info("Creating vision learner model")
        learn = vision_learner(
            dls, 
            resnet18, 
            metrics=error_rate
        )
        
        logger.info("Starting model training")
        learn.fine_tune(3, freeze_epochs=2)
        
        # Save the model
        logger.info("Saving trained model")
        learn.export('hotdog_model.pkl')
        logger.success("Model training completed successfully")
        
        return learn
        
    except Exception as e:
        logger.error("Failed to create hotdog classifier")
        logger.exception("Detailed error information:")
        raise

if __name__ == "__main__":
    logger.info("Starting hotdog classifier training process")
    
    try:
        learn = create_hotdog_classifier()
        logger.info("Training completed successfully")
            
    except Exception as e:
        logger.error("Training process failed")
        logger.exception("Detailed error information:")
        sys.exit(1) 