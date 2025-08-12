CREATE TABLE agents (
    agent_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    department VARCHAR(50),
    experience_years INT
);

CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    region VARCHAR(50)
);

CREATE TABLE tickets (
    ticket_id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id),
    agent_id INT REFERENCES agents(agent_id),
    subject VARCHAR(255),
    status VARCHAR(50),
    priority VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE ticket_events (
    event_id SERIAL PRIMARY KEY,
    ticket_id INT REFERENCES tickets(ticket_id),
    event_type VARCHAR(50),
    event_description TEXT,
    event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
