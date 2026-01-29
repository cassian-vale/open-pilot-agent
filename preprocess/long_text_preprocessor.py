import sys
from pathlib import Path
from typing import  List, Dict, Any

dir_name = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(dir_name))

from preprocess.chunk import TextChunker


# ===== 文本预处理器 =====
class LongTextPreprocessor:
    """文本预处理，负责分块和位置标记"""
    
    def __init__(self):
        self.chunker = TextChunker()
    
    def prepare_correction_chunks(self, text: str, max_chunk_length: int = 512) -> List[Dict[str, Any]]:
        """准备纠错用的文本块，确保每个块不超过最大长度限制"""
        # 首先按句子分块
        sentence_chunks = self.chunker.chunk(
            text,
            chunk_size=max_chunk_length,  # 使用最大长度作为初始分块大小
            overlap=0,
            return_sentences=True
        )
        
        # 重新组织分块，确保每个块不超过max_chunk_length
        chunks = []
        temp_chunks = []
        current_start = 0
        
        for sentence in sentence_chunks:
            # 如果当前句子为空，跳过
            if not sentence.strip():
                continue
                
            # 如果当前块加上新句子会超过限制，且当前块不为空，则保存当前块
            temp_chunk = "".join(temp_chunks)
            if temp_chunk and len(temp_chunk) + len(sentence) > max_chunk_length:
                # 保存当前块
                chunks.append({
                    "text": temp_chunk,
                    "text_start": current_start,
                    "text_end": current_start + len(temp_chunk),
                    "formatted_text": self.chunker.add_start_end(temp_chunks, start=current_start)
                })
                
                # 开始新块
                current_start += len(temp_chunk)
                temp_chunks = [sentence]
            else:
                # 添加到当前块
                temp_chunks.append(sentence)
        
        # 处理最后一个块
        if temp_chunks:
            chunks.append({
                "text": "".join(temp_chunks),
                "text_start": current_start,
                "text_end": current_start + len("".join(temp_chunks)),
                "formatted_text": self.chunker.add_start_end(temp_chunks, start=current_start)
            })
        
        return chunks
