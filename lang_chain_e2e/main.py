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

import getpass
import os

from lang_chain_e2e.utils import get_gemini_flash
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage


prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a sarcastic know it all who takes pride in mixing factual answers with subtle insults"),
        MessagesPlaceholder(variable_name="messages")
    ])

def main():
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
    llm = get_gemini_flash()
    
    print("** Type 'exit' or 'quit' to end the program")
    while (True):
        print("Query:", end=" ")
        try:
            line = input()
        except EOFError:
            break
        if line == 'exit' or line == 'quit':
            break
        chain = prompt | llm
        resp = chain.invoke({"messages": [HumanMessage(content=line)]})
        print(resp.content)