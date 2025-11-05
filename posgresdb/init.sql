-- Create tables (same as before)
CREATE TABLE Customer (
    CustomerID SERIAL PRIMARY KEY,
    Name VARCHAR(255),
    Address VARCHAR(255),
    Contact VARCHAR(255),
    Username VARCHAR(50) UNIQUE,
    Password VARCHAR(255)  -- In real apps, hash this!
);

CREATE TABLE Account (
    AccountID SERIAL PRIMARY KEY,
    CustomerID INT,
    Type VARCHAR(50),
    Balance DECIMAL(10, 2),
    FOREIGN KEY (CustomerID) REFERENCES Customer(CustomerID)
);

CREATE TABLE Transaction (
    TransactionID SERIAL PRIMARY KEY,
    AccountID INT,
    Type TEXT CHECK (Type IN ('deposit', 'withdrawal')),  -- Simulates ENUM
    Amount DECIMAL(10, 2),
    Timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (AccountID) REFERENCES Account(AccountID)
);

CREATE TABLE Beneficiary (
    BeneficiaryID SERIAL PRIMARY KEY,
    CustomerID INT,
    Name VARCHAR(255),
    AccountNumber VARCHAR(50),
    BankDetails VARCHAR(255),
    FOREIGN KEY (CustomerID) REFERENCES Customer(CustomerID)
);

-- Insert 1000 customers using generate_series
INSERT INTO Customer (Name, Address, Contact, Username, Password)
SELECT 
    'Customer ' || i,
    'Address ' || i || ', City ' || (i % 10 + 1),  -- Vary cities 1-10
    '555-' || LPAD((i * 123)::TEXT, 7, '0'),  -- Fake phone numbers
    'user' || i,
    'hashedpass' || i
FROM generate_series(1, 1000) i;

-- Insert 1-2 accounts per customer (e.g., 1500 total accounts)
INSERT INTO Account (CustomerID, Type, Balance)
SELECT 
    c.CustomerID,
    CASE WHEN RANDOM() < 0.5 THEN 'savings' ELSE 'checking' END,
    (RANDOM() * 5000 + 100)::DECIMAL(10,2)  -- Random balance $100-5100
FROM Customer c
CROSS JOIN generate_series(1, (RANDOM() * 1 + 1)::INT) a;  -- 1 or 2 accounts each

-- Insert transactions
DO $$
DECLARE
    acc RECORD;
BEGIN
    FOR acc IN SELECT AccountID FROM Account LOOP
        FOR i IN 1..(RANDOM() * 3 + 2)::INT LOOP  -- 2-5 transactions
            INSERT INTO Transaction (AccountID, Type, Amount)
            VALUES (
                acc.AccountID,
                CASE WHEN RANDOM() < 0.6 THEN 'deposit' ELSE 'withdrawal' END,
                (RANDOM() * 500 + 50)::DECIMAL(10,2)  -- $50-550
            );
        END LOOP;
    END LOOP;
END $$;

-- Insert 0-2 beneficiaries per customer
DO $$
DECLARE
    cust RECORD;
BEGIN
    FOR cust IN SELECT CustomerID FROM Customer LOOP
        FOR i IN 1..(RANDOM() * 2)::INT LOOP  -- 0-2
            INSERT INTO Beneficiary (CustomerID, Name, AccountNumber, BankDetails)
            VALUES (
                cust.CustomerID,
                'Beneficiary ' || (RANDOM() * 1000)::INT,
                'ACC' || LPAD((RANDOM() * 1000000)::TEXT, 6, '0'),
                'Bank ' || CHR(65 + (RANDOM() * 25)::INT)  -- A-Z
            );
        END LOOP;
    END LOOP;
END $$;