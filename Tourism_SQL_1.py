import pandas as pd
from sqlalchemy import create_engine
import urllib

# SQL Server config
params = urllib.parse.quote_plus(
    "DRIVER={ODBC Driver 17 for SQL Server};SERVER=LAPTOP-1T835K9N\WINCC;DATABASE=tourismDB;Trusted_Connection=yes;"
)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

# Read and export merged data
df = pd.read_sql("SELECT * FROM MergedTourismData", engine)
df.to_csv("cc",index=False)
print(" CSV exported.")
