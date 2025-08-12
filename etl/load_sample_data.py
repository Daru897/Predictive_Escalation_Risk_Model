import random #for random selection of data
from datetime import datetime, timedelta # for realistic date generation
import psycopg2 # for postgreSQL connection
from faker import Faker

fake = Faker()

# Connect to PostgreSQL database
conn = psycopg2.connect( 
    host = "db.wmziledqwcxuojbobrwm.supabase.co",
    database = 'postgres', # Default database in Supabase
    user = 'postgres', # Default user in Supabase
    password = 'T&an+Qr3FM9N8Xp',
    port = '5432') # Default port for PostgreSQL

cur = conn.cursor() #creates a cursor object to execute SQL commands    

#Insert Agents
agent_ids = []
for _ in range(10):
    cur.execute(
        "INSERT INTO agents (name, department, experience_years) VALUES(%s, %s, %s) RETURNING agent_id",
        (fake.name(), random.choice(['Sales', 'Support', 'Marketing', 'Development']), random.randint(1,20)))
    agent_ids.append(cur.fetchone()[0])
#Insert Customers
customer_ids = []
for _ in range(10):
    cur.execute(
        "INSERT INTO customers (name, email, region) VALUES(%s, %s, %s) RETURNING customer_id", 
        (fake.name(), fake.email(), random.choice(['North', 'South', 'East', 'West'])))
    customer_ids.append(cur.fetchone()[0])
#Insert Tickets
ticket_ids =[]
for _ in range(20):
    cur.execute(
        "INSERT INTO tickets (customer_id, agent_id, subject, status, priority, created_at) VALUES(%s, %s, %s, %s, %s, %s) RETURNING ticket_id",
        (
            random.choice(customer_ids),
            random.choice(agent_ids),
            fake.sentence(),
            random.choice(["Open", "In Progress", "Closed"]),
            random.choice(["Low", "Medium", "High"]),
            datetime.now() - timedelta(days=random.randint(0,90))
        )
    )
    ticket_ids.append(cur.fetchone()[0])

for ticket_id in ticket_ids:    
    for _ in range(random.randint(1,5)):
        cur.execute(
            "INSERT INTO ticket_events(ticket_id, event_type, event_description, event_time) VALUES(%s, %s, %s, %s)",
            (
                ticket_id,
                random.choice(["Created", "Updated", "Commented", "Closed"]),
                fake.text(),
                datetime.now() - timedelta(days = random.randint(0, 90))
            )
        )

conn.commit() # Commit the changes to the database
cur.close() # Close the cursor
conn.close() # Close the database connection
print("Sample data loaded successfully!") # Confirmation message
