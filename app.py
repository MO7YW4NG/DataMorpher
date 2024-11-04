import json
import networkx as nx
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
import os
from dotenv import load_dotenv
import sqlite3
import matplotlib.pyplot as plt

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 4096,
  "response_schema": content.Schema(
    type=content.Type.OBJECT,
    properties={
      "relations": content.Schema(
        type=content.Type.ARRAY,
        items=content.Schema(
          type=content.Type.OBJECT,
          required=["entity1", "entity2", "relationship"],
          properties={
            "entity1": content.Schema(
              type=content.Type.STRING,
            ),
            "entity2": content.Schema(
              type=content.Type.STRING,
            ),
            "relationship": content.Schema(
              type=content.Type.STRING,
              enum=["one-to-one", "one-to-many", "many-to-many"],
            ),
          },
        ),
      ),
      "attributes": content.Schema(
        type=content.Type.ARRAY,
        description="array of json object of the entity",
        items=content.Schema(
            type=content.Type.STRING,
        )
      ),
      "rows": content.Schema(
        type=content.Type.ARRAY,
        items=content.Schema(
          type=content.Type.STRING,
          description="insert query string, keep track on the entities"
        )
      ),
    },
    required=["relations", "attributes"]
  ),
  "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash-002",
  system_instruction="You're an expert in Database (SQL & NoSQL), Given a NoSQL JSON data, normalize it to 3NF, make sure their relations llegal, every entity is related to at least one foreign key, and every entity has at least one attribute.",
  generation_config=generation_config,
)

chat_session = model.start_chat(
  history=[
  ]
)
if os.path.exists("knowledge_graph.db"):
    os.remove("knowledge_graph.db")
# 範例 NoSQL JSON 資料
with open("data.json") as file:
    nosql_data = json.load(file)

def parse_attributes(attributes_list):
    attributes_dict = {}
    for item in attributes_list:
        try:
            table_name, columns_json = item.replace("\\","").split(': ', 1)
            columns = json.loads(columns_json)
            attributes_dict[table_name.strip()] = columns
        except ValueError as e:
            print(f"Error parsing attribute string '{item}': {e}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for table '{table_name}': {e}")
    return attributes_dict

def extract_structure_with_attributes(data):
    prompt = f"""Following is data from NoSQL:\n{json.dumps(data, indent=2)}\n\n
             Based on the relational model, normalize to 3NF and define table attributes.
             The JSON object must use the schema: 
                {{
                    "relations": [
                        {{"entity1": "<string>", "entity2": "<string>", "relationship": "<string>"}}],
                    "attributes": [{{
                        "<table_name>: {{"<column1>": "<column attribute>", "<column2>": "<column attribute>"}}",
                        "<table_name>: {{"<column1>": "<column attribute>", "<column2>": "<column attribute>"}}",
                    }}],
                    "rows": [
                        "<insert query string>",
                    ],
                }}
             """
    
    response = chat_session.send_message(prompt)

    return json.loads(response.text.strip().lower())
  
def create_knowledge_graph_from_llm_response(llm_response):
    graph = nx.DiGraph()
    lines = llm_response  # Use the dict directly

    for line in lines['relations']:
        entity1, relation, entity2 = line['entity1'].strip().upper(), line['relationship'], line['entity2'].strip().upper()
        graph.add_node(entity1)
        graph.add_node(entity2)
        graph.add_edge(entity1, entity2, relationship=relation)
    return graph

def create_tables_from_graph(graph, table_attributes, db_path='knowledge_graph.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    for node in graph.nodes():
        table_name = node.lower()
        attributes = table_attributes[table_name]
        if attributes:
            columns = ",\n".join([f"{col} {dtype.upper()}" for col, dtype in attributes.items() ])
            create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {columns}
                );
            """

        try:
            cursor.execute(create_table_query)
        except sqlite3.OperationalError as e:
            print(f"Error creating table {table_name}: {e}")
        
    conn.commit()
    conn.close()

def list_tables(db_path='knowledge_graph.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
    
    table_info = {}
    
    for table in tables:
        table_name = table[0]
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        cursor.execute(f"SELECT * FROM {table_name};")
        rows = cursor.fetchall()
        conn.close()
        table_info[table_name] = {
            "columns": [(col[1], col[2]) for col in columns],
            "rows": rows
        }
    
    return table_info

def insert(queries, db_path='knowledge_graph.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for query in queries:
      cursor.execute(query)

    
    conn.commit()
    conn.close()

def display_results(graph, tables_info):
    print("SQLite數據庫中的表及其屬性:")
    for table, info in tables_info.items():
        print(f"- {table}:")
        print("    屬性:")
        for column in info["columns"]:
            print(f"        - {column[0]}: {column[1]}")
        print("    數據:")
        for row in info["rows"]:
            row_display = ", ".join([str(item) for item in row])
            print(f"        - {row_display}")
    
    print("\n生成的知識圖譜節點和邊:")
    for node in graph.nodes():
        print(f"節點: {node}")
    
    for edge in graph.edges(data=True):
        print(f"邊: {edge[0]} <-> {edge[1]}, 關係: {edge[2]['relationship']}")
    
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color='skyblue')
    edge_labels = nx.get_edge_attributes(graph, 'relationship')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.show()
def show():
    conn = sqlite3.connect('knowledge_graph.db')
    cursor = conn.cursor()
    cursor.execute("""
                   WITH fk_info AS (
  SELECT
      json_group_array(
          json_object(
              'schema', '',  -- SQLite does not have schemas
              'table', m.name,
              'column', fk."from",
              'foreign_key_name',
                  'fk_' || m.name || '_' || fk."from" || '_' || fk."table" || '_' || fk."to",  -- Generated foreign key name
              'reference_schema', '', -- SQLite does not have schemas
              'reference_table', fk."table",
              'reference_column', fk."to",
              'fk_def',
                  'FOREIGN KEY (' || fk."from" || ') REFERENCES ' || fk."table" || '(' || fk."to" || ')' ||
                  ' ON UPDATE ' || fk.on_update || ' ON DELETE ' || fk.on_delete
          )
      ) AS fk_metadata
  FROM
      sqlite_master m
  JOIN
      pragma_foreign_key_list(m.name) fk
  ON
      m.type = 'table'
), pk_info AS (
  SELECT
      json_group_array(
          json_object(
              'schema', '',  -- SQLite does not have schemas
              'table', pk.table_name,
              'field_count', pk.field_count,
              'column', pk.pk_column,
              'pk_def', 'PRIMARY KEY (' || pk.pk_column || ')'
          )
      ) AS pk_metadata
  FROM
  (
      SELECT
          m.name AS table_name,
          COUNT(p.name) AS field_count,  -- Count of primary key columns
          GROUP_CONCAT(p.name) AS pk_column  -- Concatenated list of primary key columns
      FROM
          sqlite_master m
      JOIN
          pragma_table_info(m.name) p
      ON
          m.type = 'table' AND p.pk > 0
      GROUP BY
          m.name
  ) pk
), indexes_metadata AS (
  SELECT
      json_group_array(
          json_object(
              'schema', '',  -- SQLite does not have schemas
              'table', m.name,
              'name', idx.name,
              'column', ic.name,
              'index_type', 'B-TREE',  -- SQLite uses B-Trees for indexing
              'cardinality', '',  -- SQLite does not provide cardinality
              'size', '',  -- SQLite does not provide index size
              'unique', (CASE WHEN idx."unique" = 1 THEN 'true' ELSE 'false' END),
              'direction', '',  -- SQLite does not provide direction info
              'column_position', ic.seqno + 1  -- Adding 1 to convert from zero-based to one-based index
          )
      ) AS indexes_metadata
  FROM
      sqlite_master m
  JOIN
      pragma_index_list(m.name) idx
  ON
      m.type = 'table'
  JOIN
      pragma_index_info(idx.name) ic
), cols AS (
  SELECT
      json_group_array(
          json_object(
              'schema', '',  -- SQLite does not have schemas
              'table', m.name,
              'name', p.name,
              'type',
                  CASE
                      WHEN INSTR(LOWER(p.type), '(') > 0 THEN
                          SUBSTR(LOWER(p.type), 1, INSTR(LOWER(p.type), '(') - 1)
                      ELSE LOWER(p.type)
                  END,
              'ordinal_position', p.cid,
              'nullable', (CASE WHEN p."notnull" = 0 THEN 'true' ELSE 'false' END),
              'collation', '',
              'character_maximum_length',
                  CASE
                      WHEN LOWER(p.type) LIKE 'char%' OR LOWER(p.type) LIKE 'varchar%' THEN
                          CASE
                              WHEN INSTR(p.type, '(') > 0 THEN
                                  REPLACE(SUBSTR(p.type, INSTR(p.type, '(') + 1, LENGTH(p.type) - INSTR(p.type, '(') - 1), ')', '')
                              ELSE 'null'
                          END
                      ELSE 'null'
                  END,
              'precision',
              CASE
                  WHEN LOWER(p.type) LIKE 'decimal%' OR LOWER(p.type) LIKE 'numeric%' THEN
                      CASE
                          WHEN instr(p.type, '(') > 0 THEN
                              json_object(
                                  'precision', substr(p.type, instr(p.type, '(') + 1, instr(p.type, ',') - instr(p.type, '(') - 1),
                                  'scale', substr(p.type, instr(p.type, ',') + 1, instr(p.type, ')') - instr(p.type, ',') - 1)
                              )
                          ELSE 'null'
                      END
                  ELSE 'null'
              END,
              'default', COALESCE(REPLACE(p.dflt_value, '"', '\"'), '')
          )
      ) AS cols_metadata
  FROM
      sqlite_master m
  JOIN
      pragma_table_info(m.name) p
  ON
      m.type in ('table', 'view')
), tbls AS (
  SELECT
      json_group_array(
          json_object(
              'schema', '',  -- SQLite does not have schemas
              'table', m.name,
              'rows', -1,
              'type', 'table',
              'engine', '',  -- SQLite does not use storage engines
              'collation', ''  -- Collation information is not available
          )
      ) AS tbls_metadata
  FROM
      sqlite_master m
  WHERE
      m.type in ('table', 'view')
), views AS (
  SELECT
      json_group_array(
          json_object(
              'schema', '',
              'view_name', m.name
          )
      ) AS views_metadata
  FROM
      sqlite_master m
  WHERE
      m.type = 'view'
)
SELECT
replace(replace(replace(
      json_object(
          'fk_info', (SELECT fk_metadata FROM fk_info),
          'pk_info', (SELECT pk_metadata FROM pk_info),
          'columns', (SELECT cols_metadata FROM cols),
          'indexes', (SELECT indexes_metadata FROM indexes_metadata),
          'tables', (SELECT tbls_metadata FROM tbls),
          'views', (SELECT views_metadata FROM views),
          'database_name', 'sqlite',
          'version', sqlite_version()
      ),
      '\"', '"'),'"[', '['), ']"', ']'
) AS metadata_json_to_import;
                   """)
    a = cursor.fetchall()
    print(str(a[0]).replace("\\",""))
def main():
    structured_data = extract_structure_with_attributes(nosql_data)
    print("LLM生成的結構化信息和表屬性:\n", json.dumps(structured_data, indent=2))
    
    attributes_list = structured_data.get("attributes", [])
    table_attributes = parse_attributes(attributes_list)
    
    graph = create_knowledge_graph_from_llm_response(structured_data)
    create_tables_from_graph(graph, table_attributes)
    print("SQLite表已創建。")
    
    insert(structured_data['rows'])
    print("NoSQL 數據已插入 SQLite 表中。")
    show()
    tables_info = list_tables()
    display_results(graph, tables_info)

if __name__ == "__main__":
    # ...existing code to load data...
    with open("data.json") as file:
        nosql_data = json.load(file)
    main()