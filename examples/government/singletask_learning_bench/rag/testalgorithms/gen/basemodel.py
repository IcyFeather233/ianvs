# Copyright 2022 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division

import os
import tempfile
import time
import zipfile
import logging

import numpy as np
import random
from tqdm import tqdm
from sedna.common.config import Context
from sedna.common.class_factory import ClassType, ClassFactory
from core.common.log import LOGGER


from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto


logging.disable(logging.WARNING)

__all__ = ["BaseModel"]

os.environ['BACKEND_TYPE'] = 'TORCH'

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import torch
import hashlib
from pathlib import Path

@ClassFactory.register(ClassType.GENERAL, alias="gen")
class BaseModel:
    # RAG配置
    KNOWLEDGE_BASE_PATH = "/path/to/your/knowledge/base"  # 知识库文档路径
    VECTOR_STORE_PATH = "/path/to/vector/store"  # 向量存储路径
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 使用的embedding模型

    def __init__(self, **kwargs):
        self.model = AutoModelForCausalLM.from_pretrained(
            "/home/icyfeather/models/Qwen2-0.5B-Instruct",
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("/home/icyfeather/models/Qwen2-0.5B-Instruct")
        
        # Initialize RAG components
        self.embeddings = HuggingFaceEmbeddings(model_name=self.EMBEDDING_MODEL)
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        # 确保向量存储目录存在
        Path(self.VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
        
        # 自动初始化或加载向量存储
        self._initialize_or_load_vector_store()

    def _get_documents_hash(self):
        """计算知识库文档的哈希值，用于检测文档变化"""
        hash_md5 = hashlib.md5()
        for file in sorted(Path(self.KNOWLEDGE_BASE_PATH).glob('*.txt')):
            hash_md5.update(file.read_bytes())
            hash_md5.update(str(file.stat().st_mtime).encode())
        return hash_md5.hexdigest()

    def _save_hash(self, hash_value):
        """保存文档哈希值"""
        hash_file = Path(self.VECTOR_STORE_PATH) / "documents.hash"
        hash_file.write_text(hash_value)

    def _load_hash(self):
        """加载已存储的文档哈希值"""
        hash_file = Path(self.VECTOR_STORE_PATH) / "documents.hash"
        return hash_file.read_text() if hash_file.exists() else None

    def _initialize_or_load_vector_store(self):
        """初始化或加载向量存储，支持增量更新"""
        current_hash = self._get_documents_hash()
        stored_hash = self._load_hash()

        # 如果向量存储已存在且文档未变化，直接加载
        if stored_hash == current_hash and (Path(self.VECTOR_STORE_PATH) / "index").exists():
            LOGGER.info("Loading existing vector store...")
            self.vector_store = Chroma(
                persist_directory=self.VECTOR_STORE_PATH,
                embedding_function=self.embeddings
            )
            return

        LOGGER.info("Initializing vector store with documents...")
        # 收集所有文档
        documents = []
        for file in Path(self.KNOWLEDGE_BASE_PATH).glob('*.txt'):
            try:
                loader = TextLoader(str(file))
                documents.extend(loader.load())
            except Exception as e:
                LOGGER.warning(f"Error loading file {file}: {e}")

        if not documents:
            LOGGER.warning("No documents found in knowledge base")
            self.vector_store = None
            return

        # 分割文档并创建向量存储
        texts = self.text_splitter.split_documents(documents)
        self.vector_store = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=self.VECTOR_STORE_PATH
        )
        
        # 持久化存储
        self.vector_store.persist()
        self._save_hash(current_hash)
        LOGGER.info("Vector store initialization completed")

    def _get_relevant_context(self, query, k=3):
        """检索相关上下文"""
        if self.vector_store is None:
            LOGGER.warning("Vector store not initialized")
            return ""
            
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            LOGGER.error(f"Error retrieving context: {e}")
            return ""

    def train(self, train_data, valid_data=None, **kwargs):
        LOGGER.info("BaseModel train")
        

    def save(self, model_path):
        LOGGER.info("BaseModel save")

    def predict(self, data, input_shape=None, **kwargs):
        LOGGER.info("BaseModel predict")
        LOGGER.info(f"Dataset: {data.dataset_name}")
        LOGGER.info(f"Description: {data.description}")
        LOGGER.info(f"Data Level 1 Dim: {data.level_1_dim}")
        LOGGER.info(f"Data Level 2 Dim: {data.level_2_dim}")
        
        # 确保向量存储已初始化
        if self.vector_store is None:
            self._initialize_or_load_vector_store()
        
        answer_list = []
        for line in tqdm(data.x, desc="Processing", unit="question"):
            # 3-shot
            indices = random.sample([i for i, l in enumerate(data.x) if l != line], 3)
            history = []
            for idx in indices:
                history.append({"role": "user", "content": data.x[idx]})
                history.append({"role": "assistant", "content": data.y[idx]})
            history.append({"role": "user", "content": line})
            response = self._infer(history)
            answer_list.append(response)
        return answer_list

    def load(self, model_url=None):
        LOGGER.info("BaseModel load")

    def evaluate(self, data, model_path, **kwargs):
        LOGGER.info("BaseModel evaluate")
        
    def _infer(self, messages):
        # Get the user's query (last message)
        query = messages[-1]["content"]
        
        # Retrieve relevant context
        context = self._get_relevant_context(query)
        
        # Add context to the system message
        if context:
            system_message = {
                "role": "system", 
                "content": f"Here is some relevant context to help answer the question:\n{context}"
            }
            messages = [system_message] + messages

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
        
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            temperature = 0.1,
            top_p = 0.9
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
