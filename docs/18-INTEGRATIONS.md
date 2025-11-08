# 18. Integration Examples

## 📋 Overview

This document provides comprehensive integration examples for using Datalid 3.0 with various programming languages, frameworks, and platforms. Each example includes complete, production-ready code.

---

## 🎯 Table of Contents

- [18. Integration Examples](#18-integration-examples)
  - [📋 Overview](#-overview)
  - [🎯 Table of Contents](#-table-of-contents)
  - [🟨 JavaScript/Node.js Integration](#-javascriptnodejs-integration)
    - [Basic Node.js Client](#basic-nodejs-client)
  - [⚛️ React/Next.js Integration](#️-reactnextjs-integration)
    - [React Hook](#react-hook)
    - [React Component](#react-component)
  - [🐘 PHP Integration](#-php-integration)
  - [💎 Ruby Integration](#-ruby-integration)
  - [🔷 Go Integration](#-go-integration)
  - [🔗 Webhook Integration](#-webhook-integration)
    - [Setting Up Webhooks](#setting-up-webhooks)
  - [☁️ Cloud Platform Integration](#️-cloud-platform-integration)
    - [AWS Lambda](#aws-lambda)
    - [Google Cloud Functions](#google-cloud-functions)
  - [🗄️ Database Integration](#️-database-integration)
    - [PostgreSQL](#postgresql)
  - [📚 Related Documentation](#-related-documentation)

---

## 🟨 JavaScript/Node.js Integration

### Basic Node.js Client

```javascript
/**
 * Datalid Node.js Client
 * Complete client for interacting with Datalid API
 */

const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

class DatalidClient {
    constructor(baseURL = 'http://localhost:8000', apiKey = null) {
        this.baseURL = baseURL;
        this.apiKey = apiKey;
        
        this.client = axios.create({
            baseURL: this.baseURL,
            timeout: 30000,
            headers: apiKey ? { 'X-API-Key': apiKey } : {}
        });
    }
    
    async processImage(image, options = {}) {
        const form = new FormData();
        
        if (typeof image === 'string') {
            form.append('file', fs.createReadStream(image));
        } else if (Buffer.isBuffer(image)) {
            form.append('file', image, { filename: 'image.jpg' });
        }
        
        Object.entries(options).forEach(([key, value]) => {
            form.append(key, String(value));
        });
        
        const response = await this.client.post('/api/v1/process', form, {
            headers: form.getHeaders()
        });
        
        return response.data;
    }
    
    async processBatch(images, options = {}) {
        const form = new FormData();
        
        images.forEach((image, index) => {
            if (typeof image === 'string') {
                form.append('files', fs.createReadStream(image));
            } else if (Buffer.isBuffer(image)) {
                form.append('files', image, { filename: `image${index}.jpg` });
            }
        });
        
        Object.entries(options).forEach(([key, value]) => {
            form.append(key, String(value));
        });
        
        const response = await this.client.post('/api/v1/process/batch', form, {
            headers: form.getHeaders()
        });
        
        return response.data;
    }
}

module.exports = DatalidClient;

// Example usage
if (require.main === module) {
    (async () => {
        const client = new DatalidClient('http://localhost:8000');
        
        const result = await client.processImage('invoice.jpg', {
            ocr_engine: 'paddleocr',
            confidence: 0.3
        });
        
        console.log(`Found ${result.data.detections.length} detections`);
    })();
}
```

---

## ⚛️ React/Next.js Integration

### React Hook

```typescript
import { useState, useCallback } from 'react';
import axios from 'axios';

export function useDatalid(baseURL = 'http://localhost:8000') {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [result, setResult] = useState(null);
    
    const processImage = useCallback(async (file, options = {}) => {
        setLoading(true);
        setError(null);
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            Object.entries(options).forEach(([key, value]) => {
                formData.append(key, String(value));
            });
            
            const response = await axios.post(
                `${baseURL}/api/v1/process`,
                formData
            );
            
            setResult(response.data);
            return response.data;
            
        } catch (err) {
            setError(err);
            throw err;
        } finally {
            setLoading(false);
        }
    }, [baseURL]);
    
    return { loading, error, result, processImage };
}
```

### React Component

```typescript
import React from 'react';
import { useDatalid } from './useDatalid';

export const ImageProcessor = () => {
    const [file, setFile] = React.useState(null);
    const { loading, result, processImage } = useDatalid();
    
    const handleProcess = async () => {
        if (!file) return;
        await processImage(file, { ocr_engine: 'paddleocr' });
    };
    
    return (
        <div>
            <input
                type="file"
                onChange={(e) => setFile(e.target.files[0])}
            />
            <button onClick={handleProcess} disabled={loading}>
                {loading ? 'Processing...' : 'Process'}
            </button>
            
            {result && (
                <div>
                    <h3>Results:</h3>
                    {result.data.detections.map((det, i) => (
                        <div key={i}>
                            <p>Text: {det.text}</p>
                            <p>Confidence: {det.confidence.toFixed(2)}</p>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};
```

---

## 🐘 PHP Integration

```php
<?php
class DatalidClient {
    private $baseURL;
    private $apiKey;
    
    public function __construct($baseURL = 'http://localhost:8000', $apiKey = null) {
        $this->baseURL = rtrim($baseURL, '/');
        $this->apiKey = $apiKey;
    }
    
    public function processImage($imagePath, $options = []) {
        if (!file_exists($imagePath)) {
            throw new Exception("Image not found: $imagePath");
        }
        
        $url = $this->baseURL . '/api/v1/process';
        
        $cfile = new CURLFile($imagePath, mime_content_type($imagePath), basename($imagePath));
        
        $postData = array_merge(['file' => $cfile], $options);
        
        $ch = curl_init($url);
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_POSTFIELDS, $postData);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        
        if ($this->apiKey) {
            curl_setopt($ch, CURLOPT_HTTPHEADER, [
                'X-API-Key: ' . $this->apiKey
            ]);
        }
        
        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);
        
        if ($httpCode !== 200) {
            throw new Exception("API Error: HTTP $httpCode");
        }
        
        return json_decode($response, true);
    }
}

// Usage
$client = new DatalidClient('http://localhost:8000');
$result = $client->processImage('invoice.jpg', [
    'ocr_engine' => 'paddleocr',
    'confidence' => 0.3
]);

echo "Detections: " . count($result['data']['detections']) . "\n";
?>
```

---

## 💎 Ruby Integration

```ruby
require 'net/http'
require 'uri'
require 'json'

class DatalidClient
  def initialize(base_url = 'http://localhost:8000', api_key = nil)
    @base_url = base_url
    @api_key = api_key
  end
  
  def process_image(image_path, options = {})
    uri = URI.parse("#{@base_url}/api/v1/process")
    
    request = Net::HTTP::Post.new(uri)
    request['X-API-Key'] = @api_key if @api_key
    
    form_data = [
      ['file', File.open(image_path)]
    ]
    
    options.each do |key, value|
      form_data << [key.to_s, value.to_s]
    end
    
    request.set_form(form_data, 'multipart/form-data')
    
    response = Net::HTTP.start(uri.hostname, uri.port) do |http|
      http.request(request)
    end
    
    unless response.is_a?(Net::HTTPSuccess)
      raise "API Error: #{response.code} - #{response.body}"
    end
    
    JSON.parse(response.body)
  end
  
  def process_batch(image_paths, options = {})
    uri = URI.parse("#{@base_url}/api/v1/process/batch")
    
    request = Net::HTTP::Post.new(uri)
    request['X-API-Key'] = @api_key if @api_key
    
    form_data = image_paths.map do |path|
      ['files', File.open(path)]
    end
    
    options.each do |key, value|
      form_data << [key.to_s, value.to_s]
    end
    
    request.set_form(form_data, 'multipart/form-data')
    
    response = Net::HTTP.start(uri.hostname, uri.port) do |http|
      http.request(request)
    end
    
    JSON.parse(response.body)
  end
end

# Usage
client = DatalidClient.new('http://localhost:8000')
result = client.process_image('invoice.jpg', {
  ocr_engine: 'paddleocr',
  confidence: 0.3
})

puts "Detections: #{result['data']['detections'].length}"
```

---

## 🔷 Go Integration

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "mime/multipart"
    "net/http"
    "os"
    "path/filepath"
)

type DatalidClient struct {
    BaseURL string
    APIKey  string
    Client  *http.Client
}

type ProcessingResult struct {
    Status    string `json:"status"`
    Message   string `json:"message"`
    Timestamp string `json:"timestamp"`
    Data      struct {
        Detections []struct {
            Text       string  `json:"text"`
            Confidence float64 `json:"confidence"`
            Dates      []struct {
                Value     string `json:"value"`
                Format    string `json:"format"`
                IsExpired bool   `json:"is_expired"`
            } `json:"dates"`
        } `json:"detections"`
    } `json:"data"`
}

func NewDatalidClient(baseURL string, apiKey string) *DatalidClient {
    return &DatalidClient{
        BaseURL: baseURL,
        APIKey:  apiKey,
        Client:  &http.Client{},
    }
}

func (c *DatalidClient) ProcessImage(imagePath string, options map[string]string) (*ProcessingResult, error) {
    file, err := os.Open(imagePath)
    if err != nil {
        return nil, err
    }
    defer file.Close()
    
    body := &bytes.Buffer{}
    writer := multipart.NewWriter(body)
    
    part, err := writer.CreateFormFile("file", filepath.Base(imagePath))
    if err != nil {
        return nil, err
    }
    
    _, err = io.Copy(part, file)
    if err != nil {
        return nil, err
    }
    
    for key, value := range options {
        writer.WriteField(key, value)
    }
    
    err = writer.Close()
    if err != nil {
        return nil, err
    }
    
    url := fmt.Sprintf("%s/api/v1/process", c.BaseURL)
    req, err := http.NewRequest("POST", url, body)
    if err != nil {
        return nil, err
    }
    
    req.Header.Set("Content-Type", writer.FormDataContentType())
    if c.APIKey != "" {
        req.Header.Set("X-API-Key", c.APIKey)
    }
    
    resp, err := c.Client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        return nil, fmt.Errorf("API error: %s", resp.Status)
    }
    
    var result ProcessingResult
    err = json.NewDecoder(resp.Body).Decode(&result)
    if err != nil {
        return nil, err
    }
    
    return &result, nil
}

func main() {
    client := NewDatalidClient("http://localhost:8000", "")
    
    options := map[string]string{
        "ocr_engine": "paddleocr",
        "confidence": "0.3",
    }
    
    result, err := client.ProcessImage("invoice.jpg", options)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    
    fmt.Printf("Status: %s\n", result.Status)
    fmt.Printf("Detections: %d\n", len(result.Data.Detections))
    
    for i, det := range result.Data.Detections {
        fmt.Printf("\nDetection %d:\n", i+1)
        fmt.Printf("  Text: %s\n", det.Text)
        fmt.Printf("  Confidence: %.2f\n", det.Confidence)
    }
}
```

---

## 🔗 Webhook Integration

### Setting Up Webhooks

```python
"""
Webhook integration for async processing notifications
"""

from flask import Flask, request, jsonify
import hmac
import hashlib

app = Flask(__name__)

# Your webhook secret
WEBHOOK_SECRET = "your-webhook-secret"

@app.route('/webhook/datalid', methods=['POST'])
def datalid_webhook():
    """Handle Datalid webhook events."""
    
    # Verify signature
    signature = request.headers.get('X-Datalid-Signature')
    if not verify_signature(request.data, signature):
        return jsonify({'error': 'Invalid signature'}), 401
    
    event = request.json
    event_type = event.get('type')
    
    if event_type == 'processing.completed':
        handle_processing_completed(event['data'])
    elif event_type == 'processing.failed':
        handle_processing_failed(event['data'])
    elif event_type == 'batch.completed':
        handle_batch_completed(event['data'])
    
    return jsonify({'status': 'received'}), 200

def verify_signature(payload, signature):
    """Verify webhook signature."""
    expected = hmac.new(
        WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected)

def handle_processing_completed(data):
    """Handle completed processing."""
    print(f"Processing completed: {data['request_id']}")
    print(f"Detections: {len(data['detections'])}")
    
    # Store results in database
    # Send notification
    # Update UI

def handle_processing_failed(data):
    """Handle failed processing."""
    print(f"Processing failed: {data['request_id']}")
    print(f"Error: {data['error']}")
    
    # Log error
    # Retry or alert

def handle_batch_completed(data):
    """Handle completed batch."""
    print(f"Batch completed: {data['batch_id']}")
    print(f"Success: {data['successful']}/{data['total']}")

if __name__ == '__main__':
    app.run(port=5000)
```

---

## ☁️ Cloud Platform Integration

### AWS Lambda

```python
"""
AWS Lambda function for Datalid processing
"""

import json
import boto3
import requests
from io import BytesIO

s3 = boto3.client('s3')
datalid_api = "https://your-datalid-api.com"

def lambda_handler(event, context):
    """
    Process images uploaded to S3.
    """
    
    # Get S3 object info
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    print(f"Processing {bucket}/{key}")
    
    # Download image from S3
    response = s3.get_object(Bucket=bucket, Key=key)
    image_data = response['Body'].read()
    
    # Send to Datalid API
    files = {'file': ('image.jpg', BytesIO(image_data))}
    data = {
        'ocr_engine': 'paddleocr',
        'confidence': 0.3
    }
    
    api_response = requests.post(
        f"{datalid_api}/api/v1/process",
        files=files,
        data=data
    )
    
    result = api_response.json()
    
    # Store results in S3
    result_key = key.replace('uploads/', 'results/').replace('.jpg', '.json')
    s3.put_object(
        Bucket=bucket,
        Key=result_key,
        Body=json.dumps(result),
        ContentType='application/json'
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Processing complete',
            'detections': len(result['data']['detections'])
        })
    }
```

### Google Cloud Functions

```python
"""
Google Cloud Function for Datalid processing
"""

from google.cloud import storage
import requests
import json

def process_image(event, context):
    """
    Triggered by Cloud Storage upload.
    """
    
    file_name = event['name']
    bucket_name = event['bucket']
    
    print(f'Processing file: {file_name}')
    
    # Download image
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    image_data = blob.download_as_bytes()
    
    # Send to Datalid
    response = requests.post(
        'https://your-datalid-api.com/api/v1/process',
        files={'file': ('image.jpg', image_data)},
        data={'ocr_engine': 'paddleocr'}
    )
    
    result = response.json()
    
    # Save results
    result_blob = bucket.blob(f'results/{file_name}.json')
    result_blob.upload_from_string(
        json.dumps(result),
        content_type='application/json'
    )
    
    print(f'Found {len(result["data"]["detections"])} detections')
```

---

## 🗄️ Database Integration

### PostgreSQL

```python
"""
Store Datalid results in PostgreSQL
"""

import psycopg2
from psycopg2.extras import Json
from datetime import datetime

class DatalidDatabase:
    def __init__(self, connection_string):
        self.conn = psycopg2.connect(connection_string)
        self.create_tables()
    
    def create_tables(self):
        """Create database tables."""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS processing_results (
                    id SERIAL PRIMARY KEY,
                    request_id VARCHAR(255) UNIQUE,
                    image_path TEXT,
                    status VARCHAR(50),
                    processing_time FLOAT,
                    detections_count INTEGER,
                    result_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_request_id 
                ON processing_results(request_id);
                
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON processing_results(created_at);
            """)
            self.conn.commit()
    
    def store_result(self, result, image_path):
        """Store processing result."""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO processing_results 
                (request_id, image_path, status, processing_time, 
                 detections_count, result_data)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (request_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    result_data = EXCLUDED.result_data
            """, (
                result['request_id'],
                image_path,
                result['status'],
                result['processing_time'],
                len(result['data']['detections']),
                Json(result)
            ))
            self.conn.commit()
    
    def get_recent_results(self, limit=10):
        """Get recent processing results."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT request_id, image_path, status, 
                       detections_count, created_at
                FROM processing_results
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))
            return cur.fetchall()

# Usage
db = DatalidDatabase("postgresql://user:pass@localhost/datalid")

# Process and store
from datalid import DatalidClient
client = DatalidClient()
result = client.process_image("invoice.jpg")
db.store_result(result, "invoice.jpg")
```

---

## 📚 Related Documentation

- [16. REST API](16-API-REST.md) - Complete API reference
- [17. Python Client](17-PYTHON-CLIENT.md) - Python SDK
- [23. Examples](23-EXAMPLES.md) - More examples
- [24. Best Practices](24-BEST-PRACTICES.md) - Integration best practices

---

**Ready to integrate Datalid into your application!** 🚀
