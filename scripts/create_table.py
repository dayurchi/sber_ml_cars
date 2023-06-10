from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime
from sqlalchemy.inspection import inspect

engine = create_engine('sqlite:////app/database.db')
metadata = MetaData()

table_name = 'predictions'
table = Table(
    table_name,
    metadata,
    Column('id', Integer, primary_key=True),
    Column('session_id', String),
    Column('prediction', Integer),
    Column('date_time', DateTime),
    extend_exicting=True
)

inspector = inspect(engine)

if not inspector.has_table(table_name):
    table.create(engine)
