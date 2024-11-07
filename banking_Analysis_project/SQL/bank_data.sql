# What are the most common types of transactions (e.g., withdrawals, deposits, transfers)
select Transaction_Type , count(Transaction_Type) as count_type 
from  banking_database
group by Transaction_Type 
order by count_type desc;

# Who are the top 10 customers with the highest account balances

SELECT  first_name, Account_Balance
FROM banking_database
ORDER BY account_balance DESC
LIMIT 10;
