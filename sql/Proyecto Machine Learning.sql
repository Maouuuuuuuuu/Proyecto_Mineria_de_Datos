Create Database Proyecto_Final_MD
USE Proyecto_Final_MD

CREATE SCHEMA bronze

SELECT * FROM bronze.br_order_items 
SELECT * FROM bronze.br_products
SELECT * FROM bronze.br_orders
SELECT * FROM bronze.br_customers



---- Crear Silver -----

CREATE SCHEMA silver
 
---- Crear Tablas ---------
 
 

--- Tabla clientes ----------------------------------
CREATE TABLE silver.sl_customers
(
customer_id VARCHAR(50) NOT NULL,
zip_cpde INT,
customer_state VARCHAR(20),
load_date DATETIME DEFAULT GETDATE(),

CONSTRAINT PK_sl_customers PRIMARY KEY(customer_id)
);


--- Tabla products -----------------------------------

CREATE TABLE silver.sl_products
(
product_id VARCHAR(50) NOT NULL,
category VARCHAR(100),
load_date DATETIME DEFAULT GETDATE(),

CONSTRAINT PK_sl_products PRIMARY KEY(product_id)
);

--- Tabla orders -------------------------------------

CREATE TABLE silver.sl_orders
(
order_id VARCHAR(50) NOT NULL,
customer_id VARCHAR(50) NOT NULL,
order_status VARCHAR(30),
purchase_date DATETIME,
load_date DATETIME DEFAULT GETDATE(),

CONSTRAINT PK_sl_orders PRIMARY KEY(order_id),
CONSTRAINT FK_sl_order_customer FOREIGN KEY(customer_id) REFERENCES silver.sl_customers(customer_id)
);

--- Tabla order_items ----------------------------------

CREATE TABLE silver.sl_ordenes_items
(
order_item_id INT IDENTITY (1,1),
order_id VARCHAR(50) NOT NULL,
product_id VARCHAR(50),
price DECIMAL(18,2),
load_date DATETIME DEFAULT GETDATE(),

CONSTRAINT PK_sl_order_items PRIMARY KEY(order_item_id),
CONSTRAINT FK_order FOREIGN KEY(order_id) REFERENCES silver.sl_orders(order_id),
CONSTRAINT FK_item_product FOREIGN KEY(product_id) REFERENCES silver.sl_products(product_id),
);

--- SP llenado customers --------------------------------

INSERT INTO silver.sl_customers
SELECT DISTINCT 
customer_id,
TRY_CAST(customer_zip_code_prefix AS INT),
customer_state,
GETDATE()
FROM
bronze.br_customers

--- llenado products -------------------------------------

INSERT
INTO silver.sl_products
SELECT
product_id,
LOWER(product_category_name),
GETDATE()
FROM 
bronze.br_products

--- llenado orders ---------------------------------------

INSERT INTO silver.sl_orders
SELECT
order_id,
customer_id,
LOWER(order_status),
CAST(order_purchase_timestamp AS DATETIME),
GETDATE()
FROM
bronze.br_orders

--- llenado order items ------------------------------------

INSERT INTO silver.sl_ordenes_items
SELECT
order_id,
product_id,
price,
GETDATE()
FROM
bronze.br_order_items


---- Crear Tabla Gold----------------------------------------
 
CREATE SCHEMA gold

--- Tabla Demand ---

CREATE TABLE gold.dim_demand
(
order_id VARCHAR(50) NOT NULL,
customer_id VARCHAR(50) NOT NULL,
product_id VARCHAR(50) NOT NULL,
order_item_id INT,
price DECIMAL(10,2) NOT NULL,
order_status VARCHAR(30),
purchase_week INT,
purchase_year INT,
category VARCHAR(100),
region VARCHAR(50)

CONSTRAINT PK_dim_demand PRIMARY KEY (order_id, order_item_id),
CONSTRAINT FK_sl_order_customer FOREIGN KEY(customer_id) REFERENCES silver.sl_customers(customer_id),
CONSTRAINT FK_sl_product FOREIGN KEY(product_id) REFERENCES silver.sl_products (product_id),
CONSTRAINT FK_sl_order_items FOREIGN KEY(order_item_id) REFERENCES silver.sl_ordenes_items (order_item_id)
);

ALTER TABLE gold.dim_demand
ADD CONSTRAINT PK_dim_demand
PRIMARY KEY (order_id, order_item_id);

--- Llenado de tabla Demand ----------------------------------

INSERT INTO gold.dim_demand
(
    order_id,
    customer_id,
    product_id,
    order_item_id,
    price,
    order_status,
    category,
    purchase_year,
    purchase_week,
    region
)
SELECT
    o.order_id,
    o.customer_id,
    oi.product_id,
    oi.order_item_id,
    oi.price,
    o.order_status,
    p.category,
    YEAR(o.purchase_date) AS purchase_year,
    DATEPART(ISO_WEEK, o.purchase_date) AS purchase_week,

    CASE
        WHEN c.customer_state IN ('AC', 'AP', 'AM', 'PA', 'RO', 'RR', 'TO')
            THEN 'Norte'

        WHEN c.customer_state IN ('AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE')
            THEN 'Nordeste'

        WHEN c.customer_state IN ('DF', 'GO', 'MS', 'MT')
            THEN 'Centro-Oeste'

        WHEN c.customer_state IN ('ES', 'MG', 'RJ', 'SP')
            THEN 'Sudeste'

        WHEN c.customer_state IN ('PR', 'RS', 'SC')
            THEN 'Sul'

        ELSE 'Desconocida'
    END AS region

FROM silver.sl_orders AS o

INNER JOIN silver.sl_ordenes_items AS oi
    ON o.order_id = oi.order_id

INNER JOIN silver.sl_products AS p
    ON oi.product_id = p.product_id

INNER JOIN silver.sl_customers AS c
    ON o.customer_id = c.customer_id

WHERE o.order_status = 'delivered';


