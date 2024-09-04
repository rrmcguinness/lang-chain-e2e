# Copyright 2024 Google, LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_gemini_flash():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
def get_embeddings_model():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-004")