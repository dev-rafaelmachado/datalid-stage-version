# 17. Python Client SDK

## 📋 Overview

The Datalid Python Client SDK provides a high-level, pythonic interface for interacting with the Datalid API. It handles authentication, retries, error handling, and provides convenient methods for all API operations.

---

## 🎯 Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Client Configuration](#client-configuration)
4. [Image Processing](#image-processing)
5. [Batch Operations](#batch-operations)
6. [Async Support](#async-support)
7. [Error Handling](#error-handling)
8. [Advanced Features](#advanced-features)
9. [Examples](#examples)

---

## 📦 Installation

### From Source

```bash
# Clone repository
git clone https://github.com/your-org/datalid3.0.git
cd datalid3.0

# Install with client dependencies
pip install -e ".[client]"

# Or install specific requirements
pip install requests pillow python-dotenv
```

### Minimal Installation

```bash
# Only client dependencies
pip install requests pillow python-dotenv
```

---

## 🚀 Quick Start

### Basic Usage

```python
from src.api.client import DatalidClient

# Initialize client
client = DatalidClient(
    base_url="http://localhost:8000",
    api_key="your_api_key_here"  # Optional
)

# Process an image
result = client.process_image("path/to/image.jpg")

# Access results
for detection in result.detections:
    print(f"Found date: {detection.text}")
    for date in detection.dates:
        print(f"  - {date.value} (confidence: {date.confidence:.2f})")
```

### With Context Manager

```python
from src.api.client import DatalidClient

with DatalidClient("http://localhost:8000") as client:
    # Client automatically closes connections on exit
    result = client.process_image("image.jpg")
    print(f"Found {len(result.detections)} dates")
```

---

## ⚙️ Client Configuration

### Initialization Options

```python
from src.api.client import DatalidClient

client = DatalidClient(
    # Connection settings
    base_url="http://localhost:8000",
    api_key="your_api_key",
    timeout=30,  # Request timeout in seconds
    
    # Retry settings
    max_retries=3,
    retry_delay=1.0,  # Initial delay between retries
    retry_backoff=2.0,  # Backoff multiplier
    
    # Processing defaults
    default_ocr_engine="paddleocr",
    default_confidence=0.25,
    default_preprocessing=True,
    
    # Advanced settings
    verify_ssl=True,  # SSL certificate verification
    proxies=None,  # HTTP proxies
    headers=None,  # Additional headers
)
```

### Environment Variables

```python
# Load from environment
import os

client = DatalidClient(
    base_url=os.getenv("DATALID_API_URL", "http://localhost:8000"),
    api_key=os.getenv("DATALID_API_KEY"),
    timeout=int(os.getenv("DATALID_TIMEOUT", "30"))
)
```

### Using .env File

```bash
# .env
DATALID_API_URL=http://localhost:8000
DATALID_API_KEY=your_secret_key
DATALID_OCR_ENGINE=paddleocr
DATALID_CONFIDENCE=0.25
DATALID_TIMEOUT=30
```

```python
from dotenv import load_dotenv
from src.api.client import DatalidClient

# Load environment variables
load_dotenv()

# Client automatically picks up DATALID_* variables
client = DatalidClient.from_env()
```

---

## 🖼️ Image Processing

### Process Single Image

```python
# Basic processing
result = client.process_image("invoice.jpg")

# With options
result = client.process_image(
    "invoice.jpg",
    ocr_engine="paddleocr",
    confidence=0.3,
    preprocessing=True,
    return_debug_images=True
)

# From PIL Image
from PIL import Image

img = Image.open("photo.jpg")
result = client.process_image(img)

# From numpy array
import cv2

image_array = cv2.imread("document.jpg")
result = client.process_image(image_array)

# From bytes
with open("image.jpg", "rb") as f:
    image_bytes = f.read()
result = client.process_image(image_bytes)

# From URL
result = client.process_image_url(
    "https://example.com/invoice.jpg",
    ocr_engine="paddleocr"
)
```

### Access Results

```python
# Process image
result = client.process_image("invoice.jpg")

# Status and metadata
print(f"Status: {result.status}")
print(f"Processing time: {result.processing_time:.3f}s")
print(f"Request ID: {result.request_id}")

# Image information
print(f"Image size: {result.image_info.width}x{result.image_info.height}")
print(f"Format: {result.image_info.format}")

# Detections
print(f"\nFound {len(result.detections)} detections:")
for det in result.detections:
    print(f"\nDetection {det.id}:")
    print(f"  Class: {det.class_name}")
    print(f"  Confidence: {det.confidence:.2f}")
    print(f"  BBox: {det.bbox}")
    print(f"  Text: {det.text}")
    
    # Dates in this detection
    for date in det.dates:
        print(f"    Date: {date.value}")
        print(f"    Format: {date.format}")
        print(f"    Confidence: {date.confidence:.2f}")
        print(f"    Expired: {date.is_expired}")
        if not date.is_expired:
            print(f"    Days until expiry: {date.days_until_expiry}")

# Statistics
stats = result.statistics
print(f"\nStatistics:")
print(f"  Total detections: {stats.total_detections}")
print(f"  Valid dates: {stats.valid_dates}")
print(f"  Average confidence: {stats.average_confidence:.2f}")
print(f"  OCR engine: {stats.ocr_engine}")

# Debug images (if requested)
if result.debug_images:
    annotated_img = result.debug_images.get_annotated_image()
    annotated_img.save("annotated_output.jpg")
```

### Filter and Sort Results

```python
result = client.process_image("document.jpg")

# Filter by confidence
high_conf = [d for d in result.detections if d.confidence > 0.8]

# Filter expired dates
expired = [d for d in result.detections 
           if any(date.is_expired for date in d.dates)]

# Sort by confidence
sorted_dets = sorted(
    result.detections, 
    key=lambda d: d.confidence, 
    reverse=True
)

# Get earliest expiry date
from datetime import datetime

all_dates = []
for det in result.detections:
    for date in det.dates:
        if not date.is_expired:
            all_dates.append((date.normalized, det))

if all_dates:
    earliest = min(all_dates, key=lambda x: x[0])
    print(f"Earliest expiry: {earliest[0]}")
```

---

## 📦 Batch Operations

### Process Multiple Images

```python
from pathlib import Path

# List of image paths
image_paths = [
    "invoice1.jpg",
    "invoice2.jpg",
    "invoice3.jpg"
]

# Process batch
batch_result = client.process_batch(
    image_paths,
    ocr_engine="paddleocr",
    confidence=0.25,
    parallel=True,
    max_workers=4
)

# Access batch results
print(f"Total: {batch_result.total_images}")
print(f"Successful: {batch_result.successful}")
print(f"Failed: {batch_result.failed}")
print(f"Total time: {batch_result.total_processing_time:.2f}s")

# Process individual results
for img_result in batch_result.results:
    print(f"\n{img_result.filename}:")
    print(f"  Status: {img_result.status}")
    print(f"  Detections: {len(img_result.detections)}")
    print(f"  Time: {img_result.processing_time:.3f}s")
    
    # Handle failures
    if img_result.status == "failed":
        print(f"  Error: {img_result.error_message}")

# Batch statistics
stats = batch_result.statistics
print(f"\nBatch Statistics:")
print(f"  Total detections: {stats.total_detections}")
print(f"  Average confidence: {stats.average_confidence:.2f}")
print(f"  Average time: {stats.average_processing_time:.3f}s")
```

### Process Directory

```python
from pathlib import Path

# Process all images in directory
results = client.process_directory(
    "path/to/images",
    pattern="*.jpg",  # Glob pattern
    recursive=True,   # Include subdirectories
    ocr_engine="paddleocr",
    batch_size=10,    # Process in batches
    skip_on_error=True  # Continue on errors
)

# Iterate through results
for result in results:
    print(f"{result.filename}: {len(result.detections)} dates found")
```

### Progress Tracking

```python
from tqdm import tqdm

def progress_callback(current, total):
    """Callback for progress updates."""
    print(f"Progress: {current}/{total} ({current/total*100:.1f}%)")

# With progress callback
results = client.process_batch(
    image_paths,
    progress_callback=progress_callback
)

# Or use tqdm
results = client.process_batch(
    image_paths,
    progress_bar=True  # Uses tqdm automatically
)
```

---

## ⚡ Async Support

### Async Client

```python
import asyncio
from src.api.client import AsyncDatalidClient

async def process_images_async():
    """Process images asynchronously."""
    async with AsyncDatalidClient("http://localhost:8000") as client:
        # Single image
        result = await client.process_image("image.jpg")
        
        # Multiple images concurrently
        tasks = [
            client.process_image(f"image{i}.jpg")
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)
        
        return results

# Run async code
results = asyncio.run(process_images_async())
```

### Async Batch Processing

```python
import asyncio
from pathlib import Path

async def process_large_batch():
    """Process large batch with async."""
    async with AsyncDatalidClient("http://localhost:8000") as client:
        # Get all images
        image_paths = list(Path("images").glob("*.jpg"))
        
        # Process in chunks to avoid overwhelming server
        chunk_size = 10
        all_results = []
        
        for i in range(0, len(image_paths), chunk_size):
            chunk = image_paths[i:i+chunk_size]
            tasks = [client.process_image(img) for img in chunk]
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            all_results.extend(chunk_results)
            
            # Optional: delay between chunks
            await asyncio.sleep(0.5)
        
        return all_results

results = asyncio.run(process_large_batch())

# Handle results and errors
successful = [r for r in results if not isinstance(r, Exception)]
failed = [r for r in results if isinstance(r, Exception)]

print(f"Successful: {len(successful)}")
print(f"Failed: {len(failed)}")
```

---

## ⚠️ Error Handling

### Exception Hierarchy

```python
from src.api.client import (
    DatalidError,          # Base exception
    APIError,              # API-related errors
    ValidationError,       # Request validation errors
    ImageError,            # Image processing errors
    RateLimitError,        # Rate limit exceeded
    AuthenticationError,   # Authentication failed
    ServerError,           # Server-side errors
)
```

### Handling Specific Errors

```python
from src.api.client import (
    DatalidClient,
    ValidationError,
    RateLimitError,
    ImageError
)

client = DatalidClient("http://localhost:8000")

try:
    result = client.process_image("image.jpg", confidence=1.5)  # Invalid
    
except ValidationError as e:
    print(f"Invalid parameters: {e.message}")
    print(f"Details: {e.details}")
    
except RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after}s")
    import time
    time.sleep(e.retry_after)
    # Retry...
    
except ImageError as e:
    print(f"Image error: {e.message}")
    if e.details.get("code") == "FILE_TOO_LARGE":
        print("Try reducing image size")
        
except APIError as e:
    print(f"API error: {e.status_code} - {e.message}")
    
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Automatic Retry

```python
from src.api.client import DatalidClient

# Client with automatic retry
client = DatalidClient(
    "http://localhost:8000",
    max_retries=3,
    retry_on=[500, 502, 503, 504],  # Retry on these status codes
    retry_delay=1.0,
    retry_backoff=2.0  # Exponential backoff
)

# Retries automatically on transient errors
result = client.process_image("image.jpg")
```

### Custom Error Handler

```python
def custom_error_handler(error, attempt, max_attempts):
    """Custom error handling logic."""
    print(f"Attempt {attempt}/{max_attempts}: {error}")
    
    # Log to monitoring system
    log_error(error)
    
    # Decide whether to retry
    if isinstance(error, RateLimitError):
        return True, error.retry_after
    elif isinstance(error, ServerError):
        return True, 2 ** attempt  # Exponential backoff
    else:
        return False, 0  # Don't retry

client = DatalidClient(
    "http://localhost:8000",
    error_handler=custom_error_handler
)
```

---

## 🔧 Advanced Features

### Custom Session

```python
import requests
from src.api.client import DatalidClient

# Create custom session
session = requests.Session()
session.headers.update({"User-Agent": "MyApp/1.0"})
session.verify = "/path/to/cert.pem"  # Custom SSL cert

# Use custom session
client = DatalidClient(
    "http://localhost:8000",
    session=session
)
```

### Streaming Large Files

```python
def process_large_image(client, image_path):
    """Process large image with streaming."""
    with open(image_path, "rb") as f:
        # Stream file in chunks
        result = client.process_image(
            f,
            stream=True,
            chunk_size=8192
        )
    return result
```

### Caching Results

```python
from functools import lru_cache
import hashlib

class CachedDatalidClient(DatalidClient):
    """Client with result caching."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}
    
    def _get_image_hash(self, image_path):
        """Get hash of image file."""
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def process_image(self, image_path, **kwargs):
        """Process with caching."""
        # Generate cache key
        cache_key = (
            self._get_image_hash(image_path),
            frozenset(kwargs.items())
        )
        
        # Check cache
        if cache_key in self._cache:
            print("Returning cached result")
            return self._cache[cache_key]
        
        # Process and cache
        result = super().process_image(image_path, **kwargs)
        self._cache[cache_key] = result
        return result

# Usage
client = CachedDatalidClient("http://localhost:8000")
result1 = client.process_image("image.jpg")  # Processes
result2 = client.process_image("image.jpg")  # From cache
```

### Job Queue Integration

```python
from src.api.client import DatalidClient
from celery import Celery

app = Celery('tasks', broker='redis://localhost:6379/0')
client = DatalidClient("http://localhost:8000")

@app.task
def process_image_task(image_path, **options):
    """Celery task for image processing."""
    try:
        result = client.process_image(image_path, **options)
        return {
            "status": "success",
            "detections": len(result.detections),
            "data": result.to_dict()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# Submit task
task = process_image_task.delay("image.jpg", ocr_engine="paddleocr")
result = task.get(timeout=30)
```

### Webhook Notifications

```python
def process_with_webhook(client, image_path, webhook_url):
    """Process image and send result to webhook."""
    import requests
    
    result = client.process_image(image_path)
    
    # Send to webhook
    webhook_data = {
        "image": image_path,
        "detections": len(result.detections),
        "status": result.status,
        "timestamp": result.timestamp.isoformat()
    }
    
    requests.post(webhook_url, json=webhook_data)
    return result
```

---

## 📝 Examples

### Example 1: Invoice Processing Pipeline

```python
from src.api.client import DatalidClient
from pathlib import Path
import json

def process_invoices(invoice_dir, output_file):
    """Process all invoices and save results."""
    client = DatalidClient("http://localhost:8000")
    
    results = []
    for invoice_path in Path(invoice_dir).glob("*.jpg"):
        try:
            result = client.process_image(
                invoice_path,
                ocr_engine="paddleocr",
                confidence=0.3
            )
            
            # Extract key information
            invoice_data = {
                "filename": invoice_path.name,
                "processing_time": result.processing_time,
                "dates": []
            }
            
            for det in result.detections:
                for date in det.dates:
                    invoice_data["dates"].append({
                        "value": date.value,
                        "confidence": date.confidence,
                        "is_expired": date.is_expired,
                        "text": det.text
                    })
            
            results.append(invoice_data)
            print(f"✓ {invoice_path.name}: {len(invoice_data['dates'])} dates")
            
        except Exception as e:
            print(f"✗ {invoice_path.name}: {e}")
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nProcessed {len(results)} invoices")
    return results

# Usage
results = process_invoices("invoices/", "results.json")
```

### Example 2: Quality Control System

```python
from src.api.client import DatalidClient
from datetime import datetime, timedelta

def quality_control_check(image_path, min_confidence=0.8):
    """Check if image meets quality standards."""
    client = DatalidClient("http://localhost:8000")
    
    result = client.process_image(
        image_path,
        ocr_engine="parseq",  # High accuracy
        confidence=0.2,  # Lower threshold to catch everything
        return_debug_images=True
    )
    
    # Quality checks
    checks = {
        "has_detections": len(result.detections) > 0,
        "high_confidence": all(d.confidence >= min_confidence 
                               for d in result.detections),
        "has_valid_dates": any(d.dates for d in result.detections),
        "readable_text": all(d.text and len(d.text) > 3 
                            for d in result.detections),
    }
    
    # Overall quality
    passed = all(checks.values())
    
    # Generate report
    report = {
        "image": image_path,
        "timestamp": datetime.now().isoformat(),
        "passed": passed,
        "checks": checks,
        "detections": len(result.detections),
        "average_confidence": result.statistics.average_confidence
    }
    
    # Save debug image if failed
    if not passed and result.debug_images:
        debug_path = f"failed_{Path(image_path).name}"
        result.debug_images.save_annotated(debug_path)
        report["debug_image"] = debug_path
    
    return report

# Usage
report = quality_control_check("product_photo.jpg")
if report["passed"]:
    print("✓ Quality check passed")
else:
    print("✗ Quality issues detected:")
    for check, passed in report["checks"].items():
        if not passed:
            print(f"  - {check}")
```

### Example 3: Expiry Monitoring System

```python
from src.api.client import DatalidClient
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText

def monitor_expiry_dates(image_paths, warning_days=30):
    """Monitor products for upcoming expiry dates."""
    client = DatalidClient("http://localhost:8000")
    
    warnings = []
    
    for image_path in image_paths:
        result = client.process_image(image_path, ocr_engine="paddleocr")
        
        for det in result.detections:
            for date in det.dates:
                if not date.is_expired:
                    days_left = date.days_until_expiry
                    
                    if days_left <= warning_days:
                        warnings.append({
                            "image": image_path,
                            "date": date.value,
                            "days_left": days_left,
                            "text": det.text,
                            "bbox": det.bbox
                        })
    
    # Sort by urgency
    warnings.sort(key=lambda w: w["days_left"])
    
    # Send alert if warnings exist
    if warnings:
        send_expiry_alert(warnings)
    
    return warnings

def send_expiry_alert(warnings):
    """Send email alert for expiring products."""
    message = "Products approaching expiry:\n\n"
    
    for w in warnings:
        message += f"- {w['image']}: {w['date']} ({w['days_left']} days)\n"
    
    # Send email
    msg = MIMEText(message)
    msg['Subject'] = f"Expiry Alert: {len(warnings)} products"
    msg['From'] = "alerts@example.com"
    msg['To'] = "manager@example.com"
    
    # ... send via SMTP ...
    
    print(f"Alert sent for {len(warnings)} products")

# Usage
warnings = monitor_expiry_dates(
    ["product1.jpg", "product2.jpg", "product3.jpg"],
    warning_days=30
)
```

### Example 4: Comparative OCR Analysis

```python
from src.api.client import DatalidClient

def compare_ocr_engines(image_path, engines=None):
    """Compare results from different OCR engines."""
    if engines is None:
        engines = ["tesseract", "paddleocr", "easyocr", "parseq"]
    
    client = DatalidClient("http://localhost:8000")
    
    results = {}
    for engine in engines:
        try:
            result = client.process_image(
                image_path,
                ocr_engine=engine,
                confidence=0.25
            )
            
            results[engine] = {
                "processing_time": result.processing_time,
                "detections": len(result.detections),
                "avg_confidence": result.statistics.average_confidence,
                "dates_found": sum(len(d.dates) for d in result.detections),
                "texts": [d.text for d in result.detections]
            }
            
        except Exception as e:
            results[engine] = {"error": str(e)}
    
    # Print comparison
    print("\nOCR Engine Comparison:")
    print("=" * 60)
    for engine, data in results.items():
        print(f"\n{engine.upper()}:")
        if "error" in data:
            print(f"  Error: {data['error']}")
        else:
            print(f"  Time: {data['processing_time']:.3f}s")
            print(f"  Detections: {data['detections']}")
            print(f"  Dates: {data['dates_found']}")
            print(f"  Confidence: {data['avg_confidence']:.2f}")
    
    return results

# Usage
comparison = compare_ocr_engines("complex_document.jpg")
```

---

## 📚 Related Documentation

- [16. REST API](16-API-REST.md) - API reference
- [18. Integrations](18-INTEGRATIONS.md) - Integration examples
- [24. Best Practices](24-BEST-PRACTICES.md) - Best practices
- [22. Troubleshooting](22-TROUBLESHOOTING.md) - Common issues

---

**Next Steps:**
- [Explore API endpoints](16-API-REST.md)
- [See integration examples](18-INTEGRATIONS.md)
- [Learn best practices](24-BEST-PRACTICES.md)
