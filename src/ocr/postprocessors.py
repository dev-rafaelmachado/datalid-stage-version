"""
📅 POST-PROCESSOR 
"""
import calendar
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


class DateParser:

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._build_month_mappings()
        self._build_patterns()

    def _build_month_mappings(self):
        """Mapeamento COMPLETO de meses com fuzzy matching"""
        self.MONTHS_DIRECT = {
            # Português
            'JAN': 1, 'JANEIRO': 1,
            'FEV': 2, 'FEVEREIRO': 2,
            'MAR': 3, 'MARCO': 3, 'MARÇO': 3,
            'ABR': 4, 'ABRIL': 4,
            'MAI': 5, 'MAIO': 5,
            'JUN': 6, 'JUNHO': 6,
            'JUL': 7, 'JULHO': 7,
            'AGO': 8, 'AGOSTO': 8,
            'SET': 9, 'SETEMBRO': 9,
            'OUT': 10, 'OUTUBRO': 10,
            'NOV': 11, 'NOVEMBRO': 11,
            'DEZ': 12, 'DEZEMBRO': 12,
            # Inglês
            'JAN': 1, 'JANUARY': 1,
            'FEV': 2, 'FEB': 2, 'FEBRUARY': 2,
            'MAR': 3, 'MARCH': 3,
            'ABR': 4, 'APR': 4, 'APRIL': 4,
            'MAI': 5, 'MAY': 5,
            'JUN': 6, 'JUNE': 6,
            'JUL': 7, 'JULY': 7,
            'AGO': 8, 'AUG': 8, 'AUGUST': 8,
            'SET': 9, 'SEP': 9, 'SEPT': 9, 'SEPTEMBER': 9,
            'OUT': 10, 'OCT': 10, 'OCTOBER': 10,
            'NOV': 11, 'NOVEMBER': 11,
            'DEZ': 12, 'DEC': 12, 'DECEMBER': 12,
            # Erros OCR comuns
            'JAN0': 1, 'JANO': 1, 'FEV0': 2, 'FEVO': 2, 'MAR0': 3, 'MARO': 3,
            'ABR0': 4, 'ABRO': 4, 'MAI0': 5, 'JUN0': 6, 'JUNO': 6,
            'JUL0': 7, 'JULO': 7, 'AGO0': 8, 'AGOO': 8, 'SET0': 9, 'SETO': 9,
            'OUT0': 10, 'OUTO': 10, 'NOV0': 11, 'NOVO': 11, 'DEZ0': 12, 'DEZO': 12,
        }
        
        # Palavras-chave para contexto (EXPANDIDO v6)
        self.VALIDADE_KEYWORDS = [
            'VAL', 'VALIDADE', 'VALDADE', 'VALID', 'VENC', 'VENCIMENTO',
            'EXP', 'EXPIRY', 'EXPIRES', 'V:', 'V.', 'V-', 'VRL', 'URL', 'Y:', 'Y.', 'Y-',
            # V6: Suporte para V e Y isolados (comum em embalagens)
            ' V ', ' Y '
        ]
        
        self.FABRICACAO_KEYWORDS = [
            'FAB', 'FABRICACAO', 'FABRICAÇÃO', 'FABRICADO', 'MFG', 'MANUF',
            'F:', 'F.', 'F-', 'LOTE', 'LOT', 'L:', 'L.', 'L-', 'PROD', 'PRODUCTION'
        ]

    def _build_patterns(self):
        """Padrões organizados por TIPO e PRIORIDADE - EXPANDIDO v6"""
        
        # GRUPO 1: Padrões com PREFIXO DE VALIDADE explícito (máxima prioridade)
        self.validade_patterns = [
            # VAL: DD/MM/YYYY (v6: inclui V e Y isolados)
            {
                'regex': re.compile(
                    r'(?:VAL|VALIDADE|VALDADE|VALID|VENC|VENCIMENTO|EXP|EXPIRY|V\.|V:|V\s|Y\s|Y:|VRL|URL)[:.\-\s]*'
                    r'(\d{1,2})[/.\-\s]+(\d{1,2})[/.\-\s]+(\d{4})',
                    re.IGNORECASE
                ),
                'type': 'val_dmy_4digit',
                'handler': self._parse_dmy,
                'priority': 100
            },
            # VAL: DD/MM/YY
            {
                'regex': re.compile(
                    r'(?:VAL|VALIDADE|VALDADE|VALID|VENC|VENCIMENTO|EXP|EXPIRY|V\.|V:|V\s|Y\s|Y:|VRL|URL)[:.\-\s]*'
                    r'(\d{1,2})[/.\-\s]+(\d{1,2})[/.\-\s]+(\d{2})(?!\d)',
                    re.IGNORECASE
                ),
                'type': 'val_dmy_2digit',
                'handler': self._parse_dmy,
                'priority': 98
            },
            # VAL: DDMMMYYYY (ex: VAL15JAN26)
            {
                'regex': re.compile(
                    r'(?:VAL|VALIDADE|VALID|VENC|EXP|V\.|V:|Y:|Y\.)[:.\-\s]*'
                    r'(\d{1,2})([A-Z]{3,10})(\d{2,4})',
                    re.IGNORECASE
                ),
                'type': 'val_dmy_text',
                'handler': self._parse_dmy_text,
                'priority': 97
            },
            # VAL: DDMMYYYY compacto (EXP14042027)
            {
                'regex': re.compile(
                    r'(?:VAL|VALIDADE|VALID|VENC|EXP|V\.|V:)[:.\-\s]*'
                    r'(\d{2})(\d{2})(\d{4})',
                    re.IGNORECASE
                ),
                'type': 'val_compact8',
                'handler': self._parse_dmy,
                'priority': 96
            },
            # VAL: DDMMYY compacto
            {
                'regex': re.compile(
                    r'(?:VAL|VALIDADE|VALID|VENC|EXP|V\.|V:)[:.\-\s]*'
                    r'(\d{2})(\d{2})(\d{2})(?!\d)',
                    re.IGNORECASE
                ),
                'type': 'val_compact6',
                'handler': self._parse_dmy,
                'priority': 95
            },
            # VAL: MM/YYYY
            {
                'regex': re.compile(
                    r'(?:VAL|VALIDADE|VALID|VENC|EXP|V\.|V:)[:.\-\s]*'
                    r'(\d{1,2})[/.\-](\d{4})',
                    re.IGNORECASE
                ),
                'type': 'val_my_4digit',
                'handler': self._parse_my,
                'priority': 94
            },
            # VAL: MM/YY
            {
                'regex': re.compile(
                    r'(?:VAL|VALIDADE|VALID|VENC|EXP|V\.|V:)[:.\-\s]*'
                    r'(\d{1,2})[/.\-](\d{2})(?!\d)',
                    re.IGNORECASE
                ),
                'type': 'val_my_2digit',
                'handler': self._parse_my,
                'priority': 93
            },
            # Validade MARÇO/27
            {
                'regex': re.compile(
                    r'(?:VALIDADE|VALDADE|VALID|VENC|VENCIMENTO)[:.\-\s]*'
                    r'([A-Z]{3,10})[/.\-\s]*(\d{2,4})',
                    re.IGNORECASE
                ),
                'type': 'val_month_year',
                'handler': self._parse_my_text,
                'priority': 92
            },
            # VAL.SET25 ou VAL.MAR27
            {
                'regex': re.compile(
                    r'(?:VAL|V\.)\.?([A-Z]{3,10})(\d{2,4})',
                    re.IGNORECASE
                ),
                'type': 'val_month_year_compact',
                'handler': self._parse_my_text,
                'priority': 91
            },
        ]
        
        # GRUPO 2: Padrões SEM prefixo mas com 4 dígitos (alta prioridade)
        self.date_4digit_patterns = [
            # DD/MM/YYYY
            {
                'regex': re.compile(r'(?<!\d)(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})(?!\d)'),
                'type': 'dmy_4digit',
                'handler': self._parse_dmy,
                'priority': 85
            },
            # DDMMYYYY
            {
                'regex': re.compile(r'(?<!\d)(\d{2})(\d{2})(\d{4})(?!\d)'),
                'type': 'compact8',
                'handler': self._parse_dmy,
                'priority': 83
            },
            # MM/YYYY
            {
                'regex': re.compile(r'(?<!\d)(\d{1,2})[/.\-](\d{4})(?!\d)'),
                'type': 'my_4digit',
                'handler': self._parse_my,
                'priority': 82
            },
            # JAN/2026 ou JANEIRO/2026
            {
                'regex': re.compile(r'(?<!\w)([A-Z]{3,10})[/.\-](\d{4})(?!\d)', re.IGNORECASE),
                'type': 'month_year_4digit',
                'handler': self._parse_my_text,
                'priority': 81
            },
            # 15JAN2026
            {
                'regex': re.compile(r'(?<!\w)(\d{1,2})([A-Z]{3,10})(\d{4})(?!\d)', re.IGNORECASE),
                'type': 'dmy_text_4digit',
                'handler': self._parse_dmy_text,
                'priority': 80
            },
        ]
        
        # GRUPO 3: Padrões com 2 dígitos (menor prioridade)
        self.date_2digit_patterns = [
            # DD/MM/YY
            {
                'regex': re.compile(r'(?<!\d)(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2})(?!\d)'),
                'type': 'dmy_2digit',
                'handler': self._parse_dmy,
                'priority': 70
            },
            # DDMMYY
            {
                'regex': re.compile(r'(?<!\d)(\d{2})(\d{2})(\d{2})(?!\d)'),
                'type': 'compact6',
                'handler': self._parse_dmy,
                'priority': 68
            },
            # MM/YY
            {
                'regex': re.compile(r'(?<!\d)(\d{1,2})[/.\-](\d{2})(?!\d)'),
                'type': 'my_2digit',
                'handler': self._parse_my,
                'priority': 67
            },
            # JAN/26 ou SET25
            {
                'regex': re.compile(r'(?<!\w)([A-Z]{3,10})[/.\-]?(\d{2})(?!\d)', re.IGNORECASE),
                'type': 'month_year_2digit',
                'handler': self._parse_my_text,
                'priority': 66
            },
            # 15JAN26
            {
                'regex': re.compile(r'(?<!\w)(\d{1,2})([A-Z]{3,10})(\d{2})(?!\d)', re.IGNORECASE),
                'type': 'dmy_text_2digit',
                'handler': self._parse_dmy_text,
                'priority': 65
            },
            # DD/MMYY (formato incompleto: 13/1025)
            {
                'regex': re.compile(r'(?<!\d)(\d{1,2})[/.\-](\d{2})(\d{2})(?!\d)'),
                'type': 'dmy_incomplete',
                'handler': self._parse_dmy,
                'priority': 64
            },
        ]
        
        # GRUPO 4: Fallback patterns (v6: EXPANDIDO)
        self.fallback_patterns = [
            # 8 dígitos: DDMMYYYY ou YYYYMMDD
            {
                'regex': re.compile(r'(?<!\d)(\d{8})(?!\d)'),
                'type': 'fallback_8',
                'handler': self._parse_fallback_8,
                'priority': 30
            },
            # 6 dígitos: DDMMYY, MMYYYY ou YYMMDD
            {
                'regex': re.compile(r'(?<!\d)(\d{6})(?!\d)'),
                'type': 'fallback_6',
                'handler': self._parse_fallback_6,
                'priority': 20
            },
        ]
        
        # Combina todos os padrões
        self.all_patterns = (
            self.validade_patterns + 
            self.date_4digit_patterns + 
            self.date_2digit_patterns + 
            self.fallback_patterns
        )

    def parse(self, text: str) -> Tuple[Optional[datetime], float]:
        """Extrai data com inteligência avançada v6"""
        if not text or not text.strip():
            return None, 0.0

        original_text = text.strip()
        logger.info(f"🎯 [PARSER v7 BRUTAL] Texto: '{original_text}'")

        try:
            # 1. Limpeza ULTRA AGRESSIVA (v6)
            cleaned_text = self._cleanup_v6(original_text)
            
            # 2. Extrai TODAS as possíveis datas
            all_candidates = self._extract_all_dates(cleaned_text)
            
            if not all_candidates:
                logger.warning(f"❌ Nenhuma data encontrada")
                return None, 0.0
            
            # 3. Remove duplicatas
            all_candidates = self._remove_duplicates(all_candidates)
            
            # 4. Analisa contexto e ajusta scores (v6: com proximidade)
            all_candidates = self._analyze_context_v6(all_candidates, cleaned_text)
            
            # 5. Escolhe a melhor data com lógica ULTRA inteligente (v6)
            best_date = self._choose_best_date_v6(all_candidates, cleaned_text)
            
            if best_date:
                date_obj, score, ptype, priority, _, _ = best_date  # v6: desempacota 6 elementos
                logger.success(f"✅ Escolhido: {date_obj.strftime('%d/%m/%Y')} (score:{score:.2f} tipo:{ptype})")
                return date_obj, score
            
            return None, 0.0

        except Exception as e:
            logger.error(f"💥 ERRO: {e}", exc_info=True)
            return None, 0.0

    def _cleanup_v6(self, text: str) -> str:
        """Limpeza ULTRA AGRESSIVA v7 BRUTAL do texto"""
        cleaned = text.upper()
        
        # Correções de OCR comuns (EXPANDIDO v7)
        typo_map = {
            'VALDADE': 'VALIDADE',
            'VALIOADE': 'VALIDADE',
            'VALIDADC': 'VALIDADE',
            'VAUDADE': 'VALIDADE',
            'L0TE': 'LOTE',
            'L0T': 'LOT',
            'VA1': 'VAL',
            'V4L': 'VAL',
            'VAI': 'VAL',
            'F4B': 'FAB',
            'FAE': 'FAB',
            'FAX': 'FAB',
            'FÄ': 'FAB',  # v7: novo
            'EXF': 'EXP',
            'EXE': 'EXP',
            'URL': 'VAL',  # URL → VAL (comum em OCR)
            'UNL': 'VAL',
            'VNL': 'VAL',
            'AG0': 'AGO',  # v7: novo
            'AGOO': 'AGO',
            # Correções de dígitos (v7: expandido)
            '2O': '20',
            'O2': '02',
            'O1': '01',
            'O0': '00',
            'O3': '03',
            'O4': '04',
            'O5': '05',
            'O6': '06',
            'O7': '07',
            'O8': '08',
            'O9': '09',
            'I0': '10',
            'I1': '11',
            'I2': '12',
            'I3': '13',
            'I4': '14',
            'I5': '15',
            'I6': '16',
            'I7': '17',
            'I8': '18',
            'I9': '19',
            '5O': '50',  # v7: novo
        }
        
        for wrong, right in typo_map.items():
            cleaned = cleaned.replace(wrong, right)
        
        # v7: Separa TODAS as datas coladas com horas/minutos (BRUTAL)
        # Ex: "01/01/2603:33" → "01/01/26 03:33"
        # Ex: "07.07.2606:05" → "07.07.26 06:05"
        # Ex: "26.12.2084" → "26.12.20 84" (remove números extras)
        cleaned = re.sub(r'(\d{2}/\d{2}/\d{2,4})(\d{2}:\d{2})', r'\1 \2', cleaned)
        cleaned = re.sub(r'(\d{2}\.\d{2}\.\d{2,4})(\d{2}:\d{2})', r'\1 \2', cleaned)
        cleaned = re.sub(r'(\d{2}-\d{2}-\d{2,4})(\d{2}:\d{2})', r'\1 \2', cleaned)
        
        # v7: Separa data colada com números (ex: 24.12206 → 24.12 206)
        cleaned = re.sub(r'(\d{2})\.(\d{2})(\d{3,})', r'\1.\2 \3', cleaned)
        
        # v7: Adiciona espaços antes de palavras-chave (NOVO)
        # Ex: "L479VAL" → "L479 VAL"
        # Ex: "0003E5FAB" → "0003E5 FAB"
        for kw in ['VAL', 'VALIDADE', 'FAB', 'LOTE', 'EXP', 'VENC', 'VALID']:
            cleaned = re.sub(f'([A-Z0-9]){kw}(?=\\W|\\d|$)', rf'\1 {kw}', cleaned)
        
        # v7: Normaliza "V" e "Y" isolados (adiciona espaços ao redor)
        # Ex: "V16/06/27" → " V 16/06/27"
        cleaned = re.sub(r'(?<!\w)V(?=\d)', ' V ', cleaned)
        cleaned = re.sub(r'(?<!\w)Y(?=\d)', ' Y ', cleaned)
        
        # v7: Corrige FA: para FAB:
        cleaned = re.sub(r'\bFA:', 'FAB:', cleaned)
        
        # Remove espaços em excesso
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()

    def _extract_all_dates(self, text: str) -> List[Tuple[datetime, float, str, int, int, int]]:
        """Extrai TODAS as datas possíveis (v6: com posições para proximidade)"""
        candidates = []
        used_positions = []
        
        for pattern_info in self.all_patterns:
            pattern = pattern_info['regex']
            ptype = pattern_info['type']
            priority = pattern_info['priority']
            handler = pattern_info['handler']
            
            for match in pattern.finditer(text):
                start = match.start()
                end = match.end()
                
                # Verifica sobreposição
                if self._has_overlap(start, end, used_positions):
                    continue
                
                # Tenta parsear
                date_obj = handler(match, text)
                
                if date_obj and self._is_valid_date(date_obj):
                    # Score base depende do padrão
                    base_score = priority / 100.0
                    # v6: Agora inclui start e end position
                    candidates.append((date_obj, base_score, ptype, priority, start, end))
                    used_positions.append((start, end))
                    logger.debug(f"  ✅ {ptype}: {date_obj.strftime('%d/%m/%Y')} score:{base_score:.2f} pos:{start}-{end}")
        
        return candidates

    def _has_overlap(self, start: int, end: int, used_positions: List[Tuple[int, int]]) -> bool:
        """Verifica se há sobreposição com posições já usadas"""
        for used_start, used_end in used_positions:
            if (start < used_end and end > used_start):
                return True
        return False

    def _parse_dmy(self, match, text: str) -> Optional[datetime]:
        """Parse DD/MM/YYYY ou DD/MM/YY - v7: SEMPRE prefere 20XX"""
        try:
            groups = match.groups()
            day = int(groups[0])
            month = int(groups[1])
            year = int(groups[2])
            
            # v7: SEMPRE assume 20XX para anos de 2 dígitos (produtos têm validade futura)
            if year < 100:
                year = 2000 + year  # SEMPRE 20XX, nunca 19XX
            
            if 1 <= day <= 31 and 1 <= month <= 12 and 2020 <= year <= 2050:
                return datetime(year, month, day)
        except:
            pass
        return None

    def _parse_dmy_text(self, match, text: str) -> Optional[datetime]:
        """Parse DD MMM YYYY (ex: 15 JAN 2026) - v7: SEMPRE 20XX"""
        try:
            groups = match.groups()
            day = int(groups[0])
            month_str = groups[1]
            year = int(groups[2])
            
            month = self._parse_month(month_str)
            if not month:
                return None
            
            # v7: SEMPRE 20XX
            if year < 100:
                year = 2000 + year
            
            if 1 <= day <= 31 and 2020 <= year <= 2050:
                return datetime(year, month, day)
        except:
            pass
        return None

    def _parse_my(self, match, text: str) -> Optional[datetime]:
        """Parse MM/YYYY ou MM/YY (retorna último dia do mês) - v7: SEMPRE 20XX"""
        try:
            groups = match.groups()
            month = int(groups[0])
            year = int(groups[1])
            
            # v7: SEMPRE 20XX
            if year < 100:
                year = 2000 + year
            
            if 1 <= month <= 12 and 2020 <= year <= 2050:
                last_day = calendar.monthrange(year, month)[1]
                return datetime(year, month, last_day)
        except:
            pass
        return None

    def _parse_my_text(self, match, text: str) -> Optional[datetime]:
        """Parse MMM/YYYY ou MMM/YY (ex: JAN/2026, SET25) - v7: SEMPRE 20XX"""
        try:
            groups = match.groups()
            month_str = groups[0]
            year = int(groups[1])
            
            month = self._parse_month(month_str)
            if not month:
                return None
            
            # v7: SEMPRE 20XX
            if year < 100:
                year = 2000 + year
            
            if 2020 <= year <= 2050:
                last_day = calendar.monthrange(year, month)[1]
                return datetime(year, month, last_day)
        except:
            pass
        return None

    def _parse_fallback_8(self, match, text: str) -> Optional[datetime]:
        """Parse de 8 dígitos - v6: tenta DDMMYYYY, YYYYMMDD"""
        try:
            digits = match.group(1)
            
            # Tenta DDMMYYYY primeiro
            day, month, year = int(digits[:2]), int(digits[2:4]), int(digits[4:])
            if 1 <= day <= 31 and 1 <= month <= 12 and 2020 <= year <= 2050:
                return datetime(year, month, day)
            
            # Tenta YYYYMMDD
            year, month, day = int(digits[:4]), int(digits[4:6]), int(digits[6:])
            if 2020 <= year <= 2050 and 1 <= month <= 12 and 1 <= day <= 31:
                return datetime(year, month, day)
        except:
            pass
        return None

    def _parse_fallback_6(self, match, text: str) -> Optional[datetime]:
        """Parse de 6 dígitos - v7: SEMPRE 20XX, tenta MMYYYY, DDMMYY, YYMMDD"""
        try:
            digits = match.group(1)
            
            # v7: Tenta MMYYYY primeiro (ex: 062028 → junho/2028)
            if int(digits[:2]) <= 12:
                month, year = int(digits[:2]), int(digits[2:])
                if 1 <= month <= 12 and 2020 <= year <= 2050:
                    last_day = calendar.monthrange(year, month)[1]
                    return datetime(year, month, last_day)
            
            # Tenta DDMMYY - v7: SEMPRE 20XX
            day, month, year = int(digits[:2]), int(digits[2:4]), int(digits[4:])
            year = 2000 + year  # SEMPRE 20XX
            if 1 <= day <= 31 and 1 <= month <= 12 and 2020 <= year <= 2050:
                return datetime(year, month, day)
            
            # v7: Tenta YYMMDD - SEMPRE 20XX
            year, month, day = int(digits[:2]), int(digits[2:4]), int(digits[4:])
            year = 2000 + year  # SEMPRE 20XX
            if 2020 <= year <= 2050 and 1 <= month <= 12 and 1 <= day <= 31:
                return datetime(year, month, day)
        except:
            pass
        return None

    def _parse_month(self, month_str: str) -> Optional[int]:
        """Parse de mês com fuzzy matching"""
        if not month_str:
            return None
        
        month_str = month_str.upper().strip()
        month_str = re.sub(r'[/\.\-\s]+$', '', month_str)
        
        # Direto
        if month_str in self.MONTHS_DIRECT:
            return self.MONTHS_DIRECT[month_str]
        
        # Remove dígitos
        letters_only = re.sub(r'\d', '', month_str)
        if letters_only in self.MONTHS_DIRECT:
            return self.MONTHS_DIRECT[letters_only]
        
        # Fuzzy matching
        best_match = None
        best_ratio = 0.0
        for known_month, month_num in self.MONTHS_DIRECT.items():
            ratio = SequenceMatcher(None, month_str, known_month).ratio()
            if ratio > best_ratio and ratio > 0.65:
                best_ratio = ratio
                best_match = month_num
        
        return best_match

    def _is_valid_date(self, date: datetime) -> bool:
        """Valida se a data é realista"""
        try:
            datetime(date.year, date.month, date.day)
            return 2020 <= date.year <= 2050
        except ValueError:
            return False

    def _analyze_context_v6(self, candidates: List[Tuple], text: str) -> List[Tuple]:
        """Analisa contexto e ajusta scores - v7 BRUTAL: PENALIDADES MÁXIMAS"""
        text_upper = text.upper()
        
        # Identifica presença de palavras-chave
        has_validade = any(kw in text_upper for kw in self.VALIDADE_KEYWORDS)
        has_fabricacao = any(kw in text_upper for kw in self.FABRICACAO_KEYWORDS)
        
        # v6: Encontra posições de todas as keywords
        validade_positions = []
        for kw in self.VALIDADE_KEYWORDS:
            for match in re.finditer(re.escape(kw), text_upper):
                validade_positions.append(match.start())
        
        fabricacao_positions = []
        for kw in self.FABRICACAO_KEYWORDS:
            for match in re.finditer(re.escape(kw), text_upper):
                fabricacao_positions.append(match.start())
        
        adjusted_candidates = []
        now = datetime.now()
        
        for date_obj, score, ptype, priority, start_pos, end_pos in candidates:
            new_score = score
            
            # BOOST para padrões com prefixo de validade
            if ptype.startswith('val_'):
                new_score += 0.30  # v7: BRUTAL +0.30 (era 0.25)
                logger.debug(f"  ⬆️  {date_obj.strftime('%d/%m/%Y')}: +0.30 (prefixo VAL)")
            
            # v6: BOOST por PROXIMIDADE a keywords de validade
            if validade_positions:
                min_dist = min(abs(start_pos - vpos) for vpos in validade_positions)
                if min_dist < 10:  # Muito próximo
                    boost = 0.20  # v7: +0.20 (era 0.15)
                    new_score += boost
                    logger.debug(f"  ⬆️  {date_obj.strftime('%d/%m/%Y')}: +{boost} (proximidade VAL={min_dist})")
                elif min_dist < 20:  # Próximo
                    boost = 0.12  # v7: +0.12 (era 0.08)
                    new_score += boost
                    logger.debug(f"  ⬆️  {date_obj.strftime('%d/%m/%Y')}: +{boost} (proximidade VAL={min_dist})")
            
            # BOOST para contexto de validade
            if has_validade and not ptype.startswith('val_'):
                new_score += 0.15  # v7: +0.15 (era 0.12)
                logger.debug(f"  ⬆️  {date_obj.strftime('%d/%m/%Y')}: +0.15 (contexto VAL)")
            
            # v7: BOOST MASSIVO para datas futuras
            if date_obj > now:
                if has_validade:
                    new_score += 0.20  # v7: BRUTAL +0.20 (era 0.10)
                    logger.debug(f"  ⬆️  {date_obj.strftime('%d/%m/%Y')}: +0.20 (futuro + contexto VAL)")
                else:
                    new_score += 0.10  # v7: +0.10 para futuro sem contexto
                    logger.debug(f"  ⬆️  {date_obj.strftime('%d/%m/%Y')}: +0.10 (futuro)")
            
            # BOOST para anos de 4 dígitos
            if '4digit' in ptype:
                new_score += 0.15  # v7: BRUTAL +0.15 (era 0.07)
                logger.debug(f"  ⬆️  {date_obj.strftime('%d/%m/%Y')}: +0.15 (4 dígitos)")
            
            # v7: PENALIDADE BRUTAL por PROXIMIDADE a keywords de fabricação
            if fabricacao_positions and not validade_positions:
                min_dist = min(abs(start_pos - fpos) for fpos in fabricacao_positions)
                if min_dist < 15:
                    new_score *= 0.30  # v7: BRUTAL *0.30 (era 0.50)
                    logger.debug(f"  ⬇️  {date_obj.strftime('%d/%m/%Y')}: *0.30 (proximidade FAB={min_dist} sem VAL)")
            
            # v7: PENALIDADE BRUTAL para contexto de fabricação SEM validade
            if has_fabricacao and not has_validade:
                new_score *= 0.30  # v7: BRUTAL *0.30 (era 0.55)
                logger.debug(f"  ⬇️  {date_obj.strftime('%d/%m/%Y')}: *0.30 (contexto FAB sem VAL)")
            
            # v7: PENALIDADE BRUTAL para datas passadas
            if date_obj < now:
                if has_validade:
                    new_score *= 0.40  # v7: BRUTAL *0.40 (era 0.65)
                    logger.debug(f"  ⬇️  {date_obj.strftime('%d/%m/%Y')}: *0.40 (passado + contexto VAL)")
                else:
                    new_score *= 0.50  # v7: *0.50 para passado sem contexto
                    logger.debug(f"  ⬇️  {date_obj.strftime('%d/%m/%Y')}: *0.50 (passado)")
            
            # PENALIDADE para anos de 2 dígitos sem contexto claro
            if '2digit' in ptype and not has_validade:
                new_score *= 0.75  # v7: mais severo (era 0.85)
                logger.debug(f"  ⬇️  {date_obj.strftime('%d/%m/%Y')}: *0.75 (2 dígitos sem contexto)")
            
            # v7: PENALIDADE para padrões fallback
            if 'fallback' in ptype:
                new_score *= 0.70  # v7: mais severo (era 0.75)
                logger.debug(f"  ⬇️  {date_obj.strftime('%d/%m/%Y')}: *0.70 (fallback)")
            
            new_score = min(max(new_score, 0.0), 1.0)
            # v6: Mantém posições no retorno
            adjusted_candidates.append((date_obj, new_score, ptype, priority, start_pos, end_pos))
        
        return adjusted_candidates

    def _remove_duplicates(self, candidates: List[Tuple]) -> List[Tuple]:
        """Remove duplicatas mantendo maior score"""
        unique = {}
        
        for candidate in candidates:
            date_obj = candidate[0]
            score = candidate[1]
            priority = candidate[3]
            
            key = date_obj.strftime('%Y-%m-%d')
            
            if key not in unique:
                unique[key] = candidate
            else:
                existing_score = unique[key][1]
                existing_priority = unique[key][3]
                
                # Mantém o de maior prioridade, ou se igual, maior score
                if priority > existing_priority or (priority == existing_priority and score > existing_score):
                    unique[key] = candidate
        
        return list(unique.values())

    def _choose_best_date_v6(self, candidates: List[Tuple], text: str) -> Optional[Tuple]:
        """Escolhe a melhor data com lógica ULTRA inteligente v6"""
        if not candidates:
            return None
        
        if len(candidates) == 1:
            return candidates[0]
        
        text_upper = text.upper()
        now = datetime.now()
        
        # Ordena por score e prioridade
        candidates.sort(key=lambda x: (x[1], x[3]), reverse=True)
        
        logger.info(f"📊 Candidatos após análise v6:")
        for i, candidate in enumerate(candidates[:5], 1):
            date, score, ptype, priority = candidate[:4]
            future_str = "🔮 FUTURO" if date > now else "⏮️ PASSADO"
            logger.info(f"  {i}. {date.strftime('%d/%m/%Y')} score:{score:.3f} prior:{priority} {ptype} {future_str}")
        
        # ESTRATÉGIA 1: Se tem prefixo de validade explícito, confia nele
        val_prefixed = [c for c in candidates if c[2].startswith('val_')]
        if val_prefixed:
            # v6: Escolhe o de maior score entre os prefixados
            val_prefixed.sort(key=lambda x: x[1], reverse=True)
            logger.debug(f"  🎯 Estratégia 1: Usando padrão com prefixo VAL (melhor score)")
            return val_prefixed[0]
        
        # ESTRATÉGIA 2: Se tem contexto de validade E fabricação, escolhe a mais futura
        has_val = any(kw in text_upper for kw in self.VALIDADE_KEYWORDS)
        has_fab = any(kw in text_upper for kw in self.FABRICACAO_KEYWORDS)
        
        if has_val and has_fab:
            future_candidates = [c for c in candidates if c[0] > now]
            if future_candidates:
                logger.debug(f"  🎯 Estratégia 2: VAL+FAB presente, escolhendo mais futura")
                # v6: Entre as futuras, escolhe a de maior score
                future_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
                return future_candidates[0]
        
        # ESTRATÉGIA 3: Se só tem contexto de validade, escolhe melhor score
        if has_val and not has_fab:
            logger.debug(f"  🎯 Estratégia 3: Só VAL presente, escolhendo melhor score")
            return candidates[0]
        
        # ESTRATÉGIA 4: Prefere anos de 4 dígitos
        four_digit_candidates = [c for c in candidates if '4digit' in c[2]]
        if four_digit_candidates:
            logger.debug(f"  🎯 Estratégia 4: Preferindo ano de 4 dígitos")
            four_digit_candidates.sort(key=lambda x: x[1], reverse=True)
            return four_digit_candidates[0]
        
        # v6: ESTRATÉGIA 5: Se tem múltiplas datas, prefere a mais futura
        if len(candidates) > 1:
            future_candidates = [c for c in candidates if c[0] > now]
            if len(future_candidates) >= 2:
                logger.debug(f"  🎯 Estratégia 5: Múltiplas datas futuras, escolhendo mais distante")
                future_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
                return future_candidates[0]
        
        # ESTRATÉGIA 6: Melhor score geral
        logger.debug(f"  🎯 Estratégia 6: Usando melhor score geral")
        return candidates[0]


def test_parser():
    """Testa o parser v6 com casos reais problemáticos"""
    parser = DateParser()
    
    test_cases = [
        ("URL FEV/27 L479 00:20H1024FE", "2027-02-28", "URL → VAL FEV/27"),
        ("Lote 03/25 Validade Mar/27", "2027-03-31", "Ignora lote, pega validade"),
        ("F16/06/25 Y16/06/27 L250616", "2027-06-16", "Y = validade"),
        ("062028", "2028-06-30", "MMYYYY compacto"),
        ("Lote 04/2024 Validade Abril/26", "2026-04-30", "Texto completo"),
        ("VAL15JAN26 LOTE0003E5", "2026-01-15", "VAL prefixado"),
        ("26.12.2025 L5013 15:32", "2025-12-26", "Data padrão"),
        ("EXP140427", "2027-04-14", "EXP compacto"),
        ("U79P 12/2027 V 01/2025 F", "2027-12-31", "Múltiplas datas com V e F"),
        ("01/01/2603:33 FAB31/05/25 L5151046061", "2026-01-01", "Data colada com hora + FAB"),
        ("AB:13/05/25 VAL:10/10/25 L:250513016", "2025-10-10", "VAL vs FAB"),
        ("LOTE 09/23-B VAL.SET25", "2025-09-30", "VAL.SET25 - mês abreviado"),
    ]
    
    print("\n" + "="*100)
    print("🧪 TESTE COMPLETO DO PARSER v6 ULTRA")
    print("="*100 + "\n")
    
    passed = 0
    for test_text, expected, description in test_cases:
        print(f"\n{'─'*100}")
        print(f"📝 {description}")
        print(f"   Input: '{test_text}'")
        print(f"   Esperado: {expected}")
        
        date_obj, confidence = parser.parse(test_text)
        
        if date_obj:
            result = date_obj.strftime('%Y-%m-%d')
            print(f"   Obtido: {result} (conf: {confidence:.2%})")
            if result == expected:
                print(f"   ✅ PASSOU")
                passed += 1
            else:
                print(f"   ❌ FALHOU - Data diferente")
        else:
            print(f"   ❌ FALHOU - Nenhuma data detectada")
    
    print(f"\n{'='*100}")
    print(f"📊 Resultado: {passed}/{len(test_cases)} testes passaram ({passed/len(test_cases)*100:.1f}%)")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    test_parser()
