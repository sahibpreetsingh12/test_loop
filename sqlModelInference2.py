import os
import torch
import functools
from transformers import AutoModelForCausalLM, AutoTokenizer
from .promptTemplate import PromptTemplate
from src.core.config import settings
# from .SQLModelDownloader import SQLModelDownloader
import logging
import sqlparse
import re 
logger = logging.getLogger('fastapi')

def call_limit(threshold):
    def decorator(func):
        calls = {"count": 0}
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if calls["count"] >= threshold:
                raise Exception(f"Function call limit of {threshold} exceeded")
            calls["count"] += 1
            return func(*args, **kwargs)
        return wrapper
    return decorator

class SQLInference:
    def __init__(self):
        self.device = "cuda"

        # self.sql_model_dir = os.path.join(os.getcwd(), "sql_model")
        # self.sql_tokenizer_dir = os.path.join(os.getcwd(), "sql_tokenizer")
        
        # # Initialize SQLModelDownloader
        # model_downloader = SQLModelDownloader()
        
        # # Load or download tokenizer
        # if os.path.exists(os.path.join(self.sql_tokenizer_dir, "tokenizer_config.json")):
        #     logger.info(f"Loading tokenizer from {self.sql_tokenizer_dir}")
        #     self.sql_tokenizer = AutoTokenizer.from_pretrained(self.sql_tokenizer_dir, local_files_only=True)
        # else:
        #     logger.info("Tokenizer not found locally. Downloading...")
        #     self.sql_tokenizer = model_downloader._download_tokenizer()

        # # Load or download model
        # if os.path.exists(os.path.join(self.sql_model_dir, "config.json")):
        #     logger.info(f"Loading model from {self.sql_model_dir}")
        #     self.sql_model = AutoModelForCausalLM.from_pretrained(self.sql_model_dir, local_files_only=True , load_in_4bit=True)
        # else:
        #     logger.info("Model not found locally. Downloading...")
        #     self.sql_model = model_downloader._download_model()

        # #self.sql_model.to(self.device)
        # self.sql_model.eval()
        self.max_new_tokens = 400
        

    def generate_sql_query(self, user_query: str):
        table_schema =  self._get_table_schema()
        prompt_parameters = {"table_schema": table_schema, "user_query": user_query}
        generate_sql_query_prompt = PromptTemplate(**prompt_parameters).generate_sql_query_template()
        return  self._generate_response(generate_sql_query_prompt)

    @call_limit(threshold=settings.MAX_FEEDBACK_ATTEMPTS)
    def regenerate_sql_query(self, sql_error: str, erroneous_sql_query: str):
        table_schema =  self._get_table_schema()
        prompt_parameters = {"available_sql_query": erroneous_sql_query, "sql_error": sql_error, "table_schema": table_schema}
        regenerate_sql_query_prompt = PromptTemplate(**prompt_parameters).regenerate_error_sql_query_template()
        return self._generate_response(regenerate_sql_query_prompt)

    def _generate_response(self, prompt):
        chat = [{"role": "user", "content": prompt}]
        self.sql_tokenizer = AutoTokenizer.from_pretrained('ibm-granite/granite-3b-code-instruct')
        self.sql_model = AutoModelForCausalLM.from_pretrained('ibm-granite/granite-3b-code-instruct', device_map='cuda')
        chat = self.sql_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        input_tokens = self.sql_tokenizer(chat, return_tensors="pt").to(self.device)
        
        # with torch.no_grad():
        output = self.sql_model.generate(**input_tokens, max_new_tokens=self.max_new_tokens , num_return_sequences=1,do_sample=False,num_beams=1,temperature=0.0,top_p=1,)
        
        decoded_output = self.sql_tokenizer.batch_decode(output , skip_special_tokens=True)[0]
        sql_query = decoded_output.split("```sql")[1].split(";")[0]
        sql_query = self.extract_sql_query(sql_query)
    
        return sql_query




    def extract_sql_query(self,input_string):
        sql_keywords = [
            'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP',
            'TRUNCATE', 'MERGE', 'REPLACE', 'EXPLAIN'
        ]
        
        cleaned_string = input_string.strip()
        pattern = r'\b(' + '|'.join(sql_keywords) + r')\b'
        match = re.search(pattern, cleaned_string, re.IGNORECASE)
        
        if match:
            start_index = match.start()
            sql_query = cleaned_string[start_index:]
            
            end_match = re.search(r';|\Z', sql_query)
            if end_match:
                end_index = end_match.end()
                sql_query = sql_query[:end_index].strip()
            
            return sql_query
        else:
            return None

    def _get_table_schema(self):
        return """
        CREATE TABLE Employees (
            EmployeeID INT PRIMARY KEY,
            FirstName VARCHAR(50),
            LastName VARCHAR(50),
            Email VARCHAR(100),
            Department VARCHAR(50),
            Position VARCHAR(50),
            Salary DECIMAL(10, 2)
        );

        CREATE TABLE EmployeeDetails (
            EmployeeID INT PRIMARY KEY,
            BirthDate DATE,
            Address VARCHAR(100),
            City VARCHAR(50),
            State VARCHAR(50),
            ZipCode VARCHAR(20),
            Phone VARCHAR(20),
            FOREIGN KEY (EmployeeID) REFERENCES Employees(EmployeeID)
        );
        """