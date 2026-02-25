CREATE INDEX IF NOT EXISTS idx_user_id ON orders (user_id);
CREATE INDEX IF NOT EXISTS idx_product_id ON order_products__train (product_id);
CREATE OR REPLACE VIEW product_details AS
SELECT p.product_id, p.product_name, a.aisle, d.department
FROM products p
JOIN aisles a ON p.aisle_id = a.aisle_id
JOIN departments d ON p.department_id = d.department_id;
