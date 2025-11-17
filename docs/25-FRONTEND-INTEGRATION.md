# 🎨 Integração Frontend - Exemplos

Exemplos práticos de como integrar a API no seu frontend.

## 📋 Configuração Inicial

### React/Next.js

```javascript
// config/api.js
export const API_CONFIG = {
  // Desenvolvimento
  DEV_URL: 'http://localhost:8000',
  
  // Produção (substitua pelo seu IP/domínio)
  PROD_URL: 'http://SEU-IP-AWS:8000',
  
  // URL ativa
  BASE_URL: process.env.NODE_ENV === 'production' 
    ? 'http://SEU-IP-AWS:8000'
    : 'http://localhost:8000',
    
  // Endpoints
  ENDPOINTS: {
    PROCESS: '/process',
    HEALTH: '/health',
    INFO: '/'
  }
};
```

### Vue/Nuxt

```javascript
// plugins/api.js
export default {
  baseURL: process.env.VUE_APP_API_URL || 'http://localhost:8000',
  
  async health() {
    const response = await fetch(`${this.baseURL}/health`);
    return response.json();
  },
  
  async processImage(file, options = {}) {
    const formData = new FormData();
    formData.append('file', file);
    
    // Adicionar opções
    if (options.return_crop_images) {
      formData.append('return_crop_images', 'true');
    }
    
    const response = await fetch(`${this.baseURL}/process`, {
      method: 'POST',
      body: formData
    });
    
    return response.json();
  }
};
```

---

## 🖼️ Upload de Imagem Completo

### React com Hooks

```jsx
import { useState } from 'react';
import { API_CONFIG } from './config/api';

function ImageProcessor() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      // Validar tamanho (10MB)
      if (selectedFile.size > 10 * 1024 * 1024) {
        setError('Arquivo muito grande. Máximo 10MB.');
        return;
      }
      
      // Validar tipo
      const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
      if (!validTypes.includes(selectedFile.type)) {
        setError('Formato inválido. Use JPEG ou PNG.');
        return;
      }
      
      setFile(selectedFile);
      setError(null);
    }
  };

  const processImage = async () => {
    if (!file) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('return_crop_images', 'true');
      formData.append('return_visualization', 'true');
      
      const response = await fetch(`${API_CONFIG.BASE_URL}/process`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Erro: ${response.status}`);
      }
      
      const data = await response.json();
      setResult(data);
      
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="image-processor">
      {/* Upload */}
      <input 
        type="file" 
        accept="image/*"
        onChange={handleFileChange}
        disabled={loading}
      />
      
      {/* Botão Processar */}
      <button 
        onClick={processImage}
        disabled={!file || loading}
      >
        {loading ? 'Processando...' : 'Processar Imagem'}
      </button>
      
      {/* Erro */}
      {error && (
        <div className="error">
          ❌ {error}
        </div>
      )}
      
      {/* Resultado */}
      {result && (
        <ResultDisplay result={result} />
      )}
    </div>
  );
}

// Componente de Resultado
function ResultDisplay({ result }) {
  return (
    <div className="result">
      <h3>✅ Resultado</h3>
      
      {/* Data encontrada */}
      {result.best_date && (
        <div className="date-info">
          <h4>📅 Data de Validade</h4>
          <p className="date">{result.best_date.date}</p>
          <p>Confiança: {(result.best_date.confidence * 100).toFixed(1)}%</p>
          <p>Dias até expirar: {result.best_date.days_until_expiry}</p>
          
          {result.best_date.is_expired && (
            <p className="expired">⚠️ PRODUTO VENCIDO</p>
          )}
        </div>
      )}
      
      {/* Texto extraído */}
      {result.ocr_results?.length > 0 && (
        <div className="ocr-info">
          <h4>📝 Texto Extraído</h4>
          {result.ocr_results.map((ocr, i) => (
            <div key={i} className="ocr-item">
              <p>{ocr.text}</p>
              <small>Confiança: {(ocr.confidence * 100).toFixed(1)}%</small>
              
              {/* Crops */}
              {ocr.crop_original_base64 && (
                <div className="crops">
                  <div>
                    <h5>Original</h5>
                    <img src={ocr.crop_original_base64} alt="Crop Original" />
                  </div>
                  <div>
                    <h5>Pré-processado</h5>
                    <img src={ocr.crop_processed_base64} alt="Crop Processado" />
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
      
      {/* Visualização */}
      {result.visualization_base64 && (
        <div className="visualization">
          <h4>🎨 Visualização</h4>
          <img src={result.visualization_base64} alt="Visualização" />
        </div>
      )}
      
      {/* Segmentação */}
      {result.detections?.length > 0 && (
        <SegmentationOverlay 
          detections={result.detections}
          imageUrl={result.visualization_base64}
        />
      )}
      
      {/* Métricas */}
      <div className="metrics">
        <small>⏱️ Tempo: {result.metrics.total_time.toFixed(2)}s</small>
        <small>🔍 Detecções: {result.metrics.num_detections}</small>
      </div>
    </div>
  );
}

export default ImageProcessor;
```

---

## 🎭 Visualizar Segmentação

### Canvas Overlay

```jsx
import { useEffect, useRef } from 'react';

function SegmentationOverlay({ detections, imageUrl }) {
  const canvasRef = useRef(null);
  const imgRef = useRef(null);

  useEffect(() => {
    if (!imgRef.current || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = imgRef.current;
    
    img.onload = () => {
      // Ajustar tamanho do canvas
      canvas.width = img.width;
      canvas.height = img.height;
      
      // Desenhar imagem
      ctx.drawImage(img, 0, 0);
      
      // Desenhar segmentações
      detections.forEach(detection => {
        if (detection.segmentation) {
          ctx.beginPath();
          
          // Mover para primeiro ponto
          const [x0, y0] = detection.segmentation[0];
          ctx.moveTo(x0, y0);
          
          // Desenhar polígono
          for (let i = 1; i < detection.segmentation.length; i++) {
            const [x, y] = detection.segmentation[i];
            ctx.lineTo(x, y);
          }
          
          ctx.closePath();
          
          // Preencher com transparência
          ctx.fillStyle = 'rgba(0, 255, 0, 0.3)';
          ctx.fill();
          
          // Contorno
          ctx.strokeStyle = '#00ff00';
          ctx.lineWidth = 2;
          ctx.stroke();
          
          // Label
          ctx.fillStyle = '#00ff00';
          ctx.font = '16px Arial';
          ctx.fillText(
            `${(detection.confidence * 100).toFixed(1)}%`,
            x0,
            y0 - 10
          );
        }
      });
    };
  }, [detections, imageUrl]);

  return (
    <div style={{ position: 'relative' }}>
      <img 
        ref={imgRef}
        src={imageUrl}
        alt="Original"
        style={{ display: 'none' }}
      />
      <canvas ref={canvasRef} />
    </div>
  );
}
```

---

## ⚡ Hook Customizado

### useImageProcessor

```javascript
import { useState, useCallback } from 'react';
import { API_CONFIG } from './config/api';

export function useImageProcessor() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const processImage = useCallback(async (file, options = {}) => {
    setLoading(true);
    setError(null);
    setResult(null);
    
    try {
      // Validações
      if (file.size > 10 * 1024 * 1024) {
        throw new Error('Arquivo muito grande (máx: 10MB)');
      }
      
      // Preparar FormData
      const formData = new FormData();
      formData.append('file', file);
      
      // Opções
      Object.entries(options).forEach(([key, value]) => {
        formData.append(key, value.toString());
      });
      
      // Fazer requisição
      const response = await fetch(`${API_CONFIG.BASE_URL}/process`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Erro ao processar');
      }
      
      const data = await response.json();
      setResult(data);
      
      return data;
      
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const checkHealth = useCallback(async () => {
    try {
      const response = await fetch(`${API_CONFIG.BASE_URL}/health`);
      return response.json();
    } catch (err) {
      console.error('Health check failed:', err);
      return { status: 'unhealthy' };
    }
  }, []);

  return {
    loading,
    error,
    result,
    processImage,
    checkHealth,
  };
}

// Uso
function MyComponent() {
  const { loading, error, result, processImage } = useImageProcessor();
  
  const handleFile = async (file) => {
    try {
      await processImage(file, {
        return_crop_images: true,
        return_visualization: true,
      });
    } catch (err) {
      console.error('Erro:', err);
    }
  };
  
  // ...
}
```

---

## 🎨 Estilos (CSS)

```css
/* ImageProcessor.css */
.image-processor {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
}

.file-input {
  margin: 20px 0;
  padding: 10px;
  border: 2px dashed #ccc;
  border-radius: 8px;
  cursor: pointer;
  transition: border-color 0.3s;
}

.file-input:hover {
  border-color: #4CAF50;
}

.process-button {
  background: #4CAF50;
  color: white;
  padding: 12px 24px;
  border: none;
  border-radius: 4px;
  font-size: 16px;
  cursor: pointer;
  transition: background 0.3s;
}

.process-button:hover:not(:disabled) {
  background: #45a049;
}

.process-button:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.loading {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 20px;
  background: #f0f0f0;
  border-radius: 8px;
}

.error {
  padding: 15px;
  background: #ffebee;
  color: #c62828;
  border-radius: 8px;
  margin: 20px 0;
}

.result {
  margin-top: 30px;
  padding: 20px;
  background: #f5f5f5;
  border-radius: 8px;
}

.date-info {
  background: white;
  padding: 20px;
  border-radius: 8px;
  margin: 15px 0;
  border-left: 4px solid #4CAF50;
}

.date {
  font-size: 24px;
  font-weight: bold;
  color: #2196F3;
  margin: 10px 0;
}

.expired {
  color: #f44336;
  font-weight: bold;
  font-size: 18px;
}

.crops {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 15px;
  margin-top: 15px;
}

.crops img {
  width: 100%;
  border-radius: 4px;
  border: 2px solid #ddd;
}

.visualization img {
  width: 100%;
  max-width: 600px;
  border-radius: 8px;
  margin-top: 15px;
}

.metrics {
  display: flex;
  gap: 20px;
  margin-top: 15px;
  padding-top: 15px;
  border-top: 1px solid #ddd;
  color: #666;
}
```

---

## 🚀 Exemplo Completo - Next.js App

```jsx
// pages/index.js
import { useState } from 'react';
import ImageProcessor from '../components/ImageProcessor';
import { useImageProcessor } from '../hooks/useImageProcessor';

export default function Home() {
  const { loading, error, result, processImage, checkHealth } = useImageProcessor();
  const [healthStatus, setHealthStatus] = useState(null);

  // Verificar health ao carregar
  useEffect(() => {
    checkHealth().then(setHealthStatus);
  }, []);

  return (
    <div className="container">
      <header>
        <h1>🎯 Datalid - Detecção de Datas de Validade</h1>
        
        {healthStatus && (
          <div className={`health-badge ${healthStatus.status}`}>
            {healthStatus.status === 'healthy' ? '✅' : '❌'} 
            API: {healthStatus.status}
          </div>
        )}
      </header>

      <main>
        <ImageProcessor 
          loading={loading}
          error={error}
          result={result}
          onProcess={processImage}
        />
      </main>

      <footer>
        <p>Powered by Datalid API v3.0</p>
      </footer>
    </div>
  );
}
```

---

## 📱 Mobile (React Native)

```jsx
import React, { useState } from 'react';
import { View, Button, Image, Text, ActivityIndicator } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

const API_URL = 'http://SEU-IP-AWS:8000';

export default function App() {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      processImage(result.assets[0].uri);
    }
  };

  const processImage = async (uri) => {
    setLoading(true);
    
    try {
      const formData = new FormData();
      formData.append('file', {
        uri,
        type: 'image/jpeg',
        name: 'photo.jpg',
      });

      const response = await fetch(`${API_URL}/process`, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const data = await response.json();
      setResult(data);
      
    } catch (error) {
      alert('Erro: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={{ flex: 1, padding: 20 }}>
      <Button title="Escolher Imagem" onPress={pickImage} />
      
      {loading && <ActivityIndicator size="large" />}
      
      {image && <Image source={{ uri: image }} style={{ width: 300, height: 300 }} />}
      
      {result?.best_date && (
        <View>
          <Text>Data: {result.best_date.date}</Text>
          <Text>Confiança: {(result.best_date.confidence * 100).toFixed(1)}%</Text>
        </View>
      )}
    </View>
  );
}
```

---

## 🔧 Troubleshooting Frontend

### CORS Error

Se ver erro de CORS:

```
Access to fetch at 'http://...' from origin '...' has been blocked by CORS policy
```

**Solução no backend (API):**
```python
# src/api/main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://seu-frontend.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Network Error

```javascript
// Adicionar timeout
const controller = new AbortController();
const timeout = setTimeout(() => controller.abort(), 30000); // 30s

fetch(url, {
  signal: controller.signal
}).finally(() => clearTimeout(timeout));
```

---

**Pronto!** Agora você tem exemplos completos de integração! 🎉
