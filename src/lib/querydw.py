from decimal import Decimal
from lib.ai.modelstack import *
from lib.tools import *
from typing import Dict, List, Any
import base64
import boto3
import io
import json
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import pymysql
import traceback
matplotlib.use('Agg')  # Use non-interactive backend


def toMarkdownTable(result):
    if isinstance(result, list) and len(result) > 0:
        # First row is column names, rest is data
        columns = result[0]
        data = result[1:]
        return pd.DataFrame(data, columns=columns).to_markdown(index=False)
    else:
        return pd.DataFrame(result).to_markdown(index=False)




def convert_columns(df):
    from decimal import Decimal
    for column in df.columns:
        if df[column].apply(lambda x: isinstance(x, (int, float, Decimal))).all():
            df[column] = df[column].astype(float)
    return df


def generate_chart(data, viz_config, pythoncode):
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data[1:], columns=data[0])
        convert_columns(df)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # pythoncode is a string of python code that should be executed dynamically here.
        exec(pythoncode)

        # Improve layout
        plt.tight_layout()
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        
        return f"data:image/png;base64,{image_base64}"
    
    except Exception as e:
        print(f"Error generating chart: {e}")
        plt.close('all')
        return None




class SqlQuery:
    def __init__(self):
        pass

    def query(self, sql):
        # Query the database
        raise NotImplementedError("Subclasses must implement this method.")



class MySqlQuery(SqlQuery):
    def __init__(self, credentials):
        self.credentials = credentials
        self.sysprompt = readText(findPath('data/system-prompt-dw.txt'))


    def query(self, sql, args = None):
        # Query the MySQL database
        connection = pymysql.connect(host=self.credentials['host'], user=self.credentials['user'], password=self.credentials['password'], database=self.credentials['database'])
        cursor = connection.cursor()
        try:
            cursor.execute(sql, args)
            cursor.connection.commit()
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            return None
        if cursor.description:
            columns = [desc[0] for desc in cursor.description]
        else:
            columns = []
        if cursor.rowcount > 0:
            result = cursor.fetchall()
            results = [columns] + [list(row) for row in result]
        else:
            results = []
        connection.close()
        # Return list with column names as first item, followed by data rows
        return results


    def insert_log(self, key, value, user):
        """
        Log the query and response to the database.
        """
        # print(f"Logging query and response to the database: {key} {value} {user}")
        sql = "INSERT INTO log (`key`, `value`, `user`) VALUES (%s, %s, %s)"
        args = (key, value, user)
        self.query(sql, args)
    




class QueryDW:
    def __init__(self, dw: SqlQuery, model_stack: ModelStack):
        self.dw = dw
        self.model_stack = model_stack
        self.context_window = from_metric(model_stack.config['context-window'])

    def query_mysql(self, sql):
        return self.model_stack.query(sql)


    def query(self, query):
        sysprompt = readText(findPath('data/system-prompt-dw.txt'))

        # Get the length of the body so we can truncate it if it's too long.
        nchars = len(sysprompt)
        tokens = nchars / 4
        cw = int(self.context_window * 0.9)
        if tokens > cw:
            print(f"Warning: Input is too long for requested model. Truncating to {200000} tokens.")
            body = body[:cw * 4]

        prompt = sysprompt + "\n\n" + query

        for i in range(5):
            error = None
            try:
                j = self.model_stack.query(prompt)
                if not j:
                    break
                j = clean_json(j)
                j = json.loads(j)
                if 'sql' in j:
                    sql = j['sql']
                    sql = sql.replace(' FROM ', '\nFROM ')
                    sql = sql.replace(' AND ', '\n  AND ')
                    sql = sql.replace(' JOIN ', '\nJOIN ')
                    sql = sql.replace(' LEFT JOIN ', '\nLEFT JOIN ')
                    sql = sql.replace(' RIGHT JOIN ', '\nRIGHT JOIN ')
                    sql = sql.replace(' FULL JOIN ', '\nFULL JOIN ')
                    sql = sql.replace(' INNER JOIN ', '\nINNER JOIN ')
                    sql = sql.replace(' OUTER JOIN ', '\nOUTER JOIN ')
                    sql = sql.replace(' CROSS JOIN ', '\nCROSS JOIN ')
                    sql = sql.replace(' NATURAL JOIN ', '\nNATURAL JOIN ')
                    sql = sql.replace(' SELF JOIN ', '\nSELF JOIN ')
                    sql = sql.replace(' UNNEST ', '\nUNNEST ')
                    sql = sql.replace(' ARRAY_AGG ', '\nARRAY_AGG ')
                    sql = sql.replace(' WHERE ', '\nWHERE ')
                    sql = sql.replace(' GROUP BY ', '\nGROUP BY ')
                    sql = sql.replace(' ORDER BY ', '\nORDER BY ')
                    sql = sql.replace(' LIMIT ', '\nLIMIT ')
                    sql = sql.replace(' OFFSET ', '\nOFFSET ')
                    sql = sql.replace(' HAVING ', '\nHAVING ')
                    sql = sql.replace(' UNION ', '\nUNION ')
                    sql = sql.replace(' INTERSECT ', '\nINTERSECT ')
                    sql = sql.replace(' EXCEPT ', '\nEXCEPT ')
                    table = self.dw.query(sql)

                    # Check if visualizations are requested
                    visualizations = j.get('visualizations', [])
                    
                    table_example = table[:10]
                    p1 = prompt + "\n\nSQL:\n\n" + sql + "\n\nRESULTS:\n\n" + toMarkdownTable(table_example) + "\n\nExplain the results of this query in a few sentences and why the SQL was correct."
                    answer = self.query_mysql(p1)
                    
                    # Build response with visualizations if present
                    response = f"""# Prompt
{query}

# SQL
```sql
{sql}
```

# Results
{toMarkdownTable(table)}
"""
                    
                    # Generate visualizations
                    if visualizations:
                        response += "\n# Visualizations\n\n"
                        for viz in visualizations:
                            chart_data = generate_chart(table, viz, viz['pythoncode'])
                            if chart_data:
                                response += f'<img src="{chart_data}" class="img-fluid my-3" alt="{viz.get("title", "Chart")}" style="max-width: 100%; height: auto;"/>\n\n'
                    
                    response += f"""
# Explanation
{answer}
"""
                    return response
                    
                elif 'error' in j:
                    error = j['error']
                    question = j['question']
                    return f"{error}\n\n{question}"

            except Exception as e:
                print(f"Error: {e}")
                tb = traceback.format_exc()
                error = e
                prompt = prompt + "\n\nERROR: " + str(e) + "\n\n" + tb
                raise e

        if error:
            p1 = prompt + "\n\nSQL:\n\n" + sql + "\n\nERROR:\n\n" + str(error) + "\n\nExplain the final error and why the SQL was incorrect."
            answer = self.model_stack.query(p1)
            answer = f"""
PROMPT: 
{prompt}

ERROR:
{error}

EXPLANATION:
{answer}
            """
        return answer


    






def test1():
    credentials = getYaml('credentials')
    sqlcreds = credentials['prod']['databases']['mysql']['zycloan']
    dw = MySqlQuery(sqlcreds)
    llm = ModelStack.from_config(credentials['modelstack']['bedrock-claude-opus-4-5'])
    qdw = QueryDW(dw, llm)

    result = qdw.query("How many loans did we make that came from August campaign?")
    print(result)

    result = qdw.query("How many cows were sold to France?")
    print(result)


if __name__ == "__main__":
    test1()
