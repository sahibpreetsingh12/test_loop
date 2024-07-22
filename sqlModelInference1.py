import torch, os, functools,logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.promptTemplate import PromptTemplate
from src.utils.SQLModelDownloader import SQLModelDownloader
import requests,json
logger = logging.getLogger('fastapi')

# Call limiter decorator
def call_limit(threshold):
    def decorator(func):
        # Track the number of calls
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
        from src.core.config import settings
        # Initialize SQLModelDownloader
        model_downloader = SQLModelDownloader()

        self.device = settings.HARDWARE_DEVICE

        potential_path = os.getcwd()+'/'

        if os.path.exists(os.path.join(potential_path, "sql_model/config.json")):
          
          self.sql_model_dir = os.path.join(potential_path, "sql_model/")
          logger.info(f"Loading Model from {self.sql_model_dir}")
          self.sql_model = AutoModelForCausalLM.from_pretrained(self.sql_model_dir, local_files_only=True )
        else:

          logger.info("Model not found locally. Downloading...")
          self.sql_model = model_downloader._download_model()


        if os.path.exists(os.path.join(potential_path, "sql_tokenizer/tokenizer_config.json")):
           
           self.sql_tokenizer_dir = os.path.join(potential_path, "sql_tokenizer/")
           logger.info(f"Loading Tokenizer from {self.sql_tokenizer_dir}")
           self.sql_tokenizer = AutoTokenizer.from_pretrained(self.sql_tokenizer_dir, local_files_only=True)

        else:
           logger.info("Tokenizer not found locally. Downloading...")
           self.sql_tokenizer = model_downloader._download_tokenizer()

        self.sql_model.to(self.device)
        self.max_new_tokens = 200

        

    def generate_sql_query(self,user_query:str):
        db_name = "string"
        connection_url = "mysql://admin:passw0rd!1@54.158.186.247:3306/genai"
        table_schema = self.get_table_schema_from_api(db_name, connection_url)
        
        # table_schema = self._get_table_schema()          # TODO: Call from SQL Connector to MongoDB endpoint
        _prompt_parameters = {"table_schema":table_schema,"user_query":user_query}
        prompt_template = PromptTemplate(**_prompt_parameters)
        generate_sql_query_prompt = prompt_template.generate_sql_query_template()
        chat = [{"role":"user", "content":generate_sql_query_prompt}]
        print('chat for generate is -->',chat)
        chat = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        input_tokens = self.tokenizer(chat, return_tensors="pt")
        for i in input_tokens:
            input_tokens[i] = input_tokens[i].to(self.device)
        output = self.sql_model.generate(**input_tokens, max_new_tokens=self.max_new_tokens)
        output = self.sql_tokenizer.batch_decode(output)
        sql_query = output                                          # TODO:print(i) for i in output: Test the output on AWS instance or Google Colab
        return sql_query
    
    @call_limit(threshold=3)                                # TODO: Make hardcoded as env varible config
    def regenerate_sql_query(self,sql_error:str,erroneous_sql_query:str):  
        db_name = "string"
        connection_url = "mysql://admin:passw0rd!1@54.158.186.247:3306/genai"
        table_schema = self.get_table_schema_from_api(db_name, connection_url)  

        # table_schema = self._get_table_schema()               # TODO: Call from SQL Connector to obtain table schema. This will be replaced with vectorDB reranker endpoint in v2.
        _prompt_parameters = {"available_sql_query":erroneous_sql_query,"sql_error":sql_error, "table_schema":table_schema}
        # regenerate_sql_query_prompt = PromptTemplate.regenerate_error_sql_query_template(**_prompt_parameters)
        prompt_template = PromptTemplate(**_prompt_parameters)
        regenerate_sql_query_prompt = prompt_template.regenerate_error_sql_query_template()
        
        chat = [{"role":"user", "content":regenerate_sql_query_prompt}]
        print('chat for regenrate is ',regenerate_sql_query_prompt)

        chat = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        input_tokens = self.tokenizer(chat, return_tensors="pt")
        for i in input_tokens:
            input_tokens[i] = input_tokens[i].to(self.device)
        output = self.sql_model.generate(**input_tokens, max_new_tokens=self.max_new_tokens)
        output = self.sql_tokenizer.batch_decode(output)
        updated_sql_query = output  
        # updated_sql_query = updated_sql_query[0].split("```sql")[2].split("\n")[1]      
        return updated_sql_query
    

    def get_table_schema_from_api(self,db_name, connection_url):
        api_url = 'https://api.genx.ayssoftwaresolution.com/sql-connector/api/v1/fetch-metadata'
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        payload = json.dumps([
            {
                "dbname": db_name,
                "connectionUrl": connection_url
            }
        ])

        response = requests.post(api_url, headers=headers, data=payload)
        if response.status_code == 200:
            data = response.json()
            schema = ""
            for table_info in data[db_name]:
                table_name = table_info['table_name']
                schema += f"CREATE TABLE {table_name} (\n"
                
                # Add columns
                for column in table_info['columns']:
                    nullable = "NULL" if column['nullable'] else "NOT NULL"
                    default = f"DEFAULT {column['default']}" if column['default'] else ""
                    schema += f"    {column['name']} {column['type']} {nullable} {default},\n"
                
                # Add primary key
                if table_info['primary_keys']:
                    primary_keys = ", ".join(table_info['primary_keys'])
                    schema += f"    PRIMARY KEY ({primary_keys}),\n"
                
                # Add foreign keys
                for fk in table_info['foreign_keys']:
                    schema += f"    FOREIGN KEY ({fk['column']}) REFERENCES {fk['referenced_table']}({fk['referenced_column']}),\n"
                
                # Remove trailing comma and close the table definition
                schema = schema.rstrip(',\n') + "\n);\n\n"
            
            return schema.strip()
        else:
            return f"Error: {response.status_code} - {response.text}"