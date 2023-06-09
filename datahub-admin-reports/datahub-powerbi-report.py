# Databricks notebook source
# MAGIC %sql 
# MAGIC
# MAGIC drop table if exists project_users;
# MAGIC drop table if exists project_costs;
# MAGIC drop table if exists projects;
# MAGIC drop table if exists project_credits;
# MAGIC drop table if exists portalusers;
# MAGIC
# MAGIC

# COMMAND ----------

spark.sql(
"CREATE TABLE project_users " +
"USING sqlserver OPTIONS ("+
  "dbtable 'dbo.project_users'," +
  "host 'fsdh-portal-sql-poc.database.windows.net',"+
  "port '1433',"+
  "database 'dh-portal-projectdb',"+
  "user 'fsdh-portal-sqlreadonly',"+
  "password '"+ 
  spark.conf.get('spark.readonly_password') + # CONFIG: spark.readonly_password {{secrets/datahub/datahub-mssql-readonly-password}}
  "');");
  
spark.sql(
"CREATE TABLE project_costs " +
"USING sqlserver OPTIONS ("+
  "dbtable 'dbo.project_costs'," +
  "host 'fsdh-portal-sql-poc.database.windows.net',"+
  "port '1433',"+
  "database 'dh-portal-projectdb',"+
  "user 'fsdh-portal-sqlreadonly',"+
  "password '"+ 
  spark.conf.get('spark.readonly_password') + # CONFIG: spark.readonly_password {{secrets/datahub/datahub-mssql-readonly-password}}
  "');");

spark.sql(
"CREATE TABLE projects " +
"USING sqlserver OPTIONS ("+
  "dbtable 'dbo.projects'," +
  "host 'fsdh-portal-sql-poc.database.windows.net',"+
  "port '1433',"+
  "database 'dh-portal-projectdb',"+
  "user 'fsdh-portal-sqlreadonly',"+
  "password '"+ 
  spark.conf.get('spark.readonly_password') + # CONFIG: spark.readonly_password {{secrets/datahub/datahub-mssql-readonly-password}}
  "');");

spark.sql(
"CREATE TABLE project_credits " +
"USING sqlserver OPTIONS ("+
  "dbtable 'dbo.project_credits'," +
  "host 'fsdh-portal-sql-poc.database.windows.net',"+
  "port '1433',"+
  "database 'dh-portal-projectdb',"+
  "user 'fsdh-portal-sqlreadonly',"+
  "password '"+ 
  spark.conf.get('spark.readonly_password') + # CONFIG: spark.readonly_password {{secrets/datahub/datahub-mssql-readonly-password}}
  "');")

spark.sql(
"CREATE TABLE portalusers " +
"USING sqlserver OPTIONS ("+
  "dbtable 'dbo.portalusers'," +
  "host 'fsdh-portal-sql-poc.database.windows.net',"+
  "port '1433',"+
  "database 'dh-portal-projectdb',"+
  "user 'fsdh-portal-sqlreadonly',"+
  "password '"+ 
  spark.conf.get('spark.readonly_password') + # CONFIG: spark.readonly_password {{secrets/datahub/datahub-mssql-readonly-password}}
  "');");


# COMMAND ----------

# MAGIC
# MAGIC  %sql
# MAGIC
# MAGIC  create or replace view project_user_count as
# MAGIC  select dept, count(9) as user_count from (
# MAGIC  select case lower(substring(domain, 1, charindex('.', domain)-1)) when 'agr' then 'AAFC' when 'asc-csa'  then 'CSA' when 'cnrc-nrc' then 'NRC' when 'dfo-mpo' then 'DFO' when 'ec' then 'ECCC' when 'inspection' then 'CFIA' when 'nrcan' then 'NRCan' when 'nrcan-rncan' then 'NRCan' when 'nrc-cnrc' then 'NRC'when 'otc-cta' then 'CTA' when 'ssc-spc' then 'SSC' else domain end as dept from (
# MAGIC  SELECT RIGHT (user_name,
# MAGIC  LEN(user_name) - CHARINDEX( '@', user_name)) AS Domain
# MAGIC  FROM project_users) as emails where domain not like 'apption%') as departments group by dept;
# MAGIC
# MAGIC
# MAGIC  create or replace view project_workspace_count as 
# MAGIC  select dept, count(9) workspace_count from (
# MAGIC  select Project_Acronym_CD, case lower(substring(domain, 1, charindex('.', domain)-1)) when 'agr' then 'AAFC' when 'asc-csa'  then 'CSA' when 'cnrc-nrc' then 'NRC' when 'dfo-mpo' then 'DFO' when 'ec' then 'ECCC' when 'inspection' then 'CFIA' when 'nrcan' then 'NRCan' when 'nrcan-rncan' then 'NRCan' when 'nrc-cnrc' then 'NRC'when 'otc-cta' then 'CTA' when 'ssc-spc' then 'SSC' else domain end as dept
# MAGIC  from
# MAGIC  (select distinct Project_Acronym_CD, RIGHT (user_name,LEN(user_name) - CHARINDEX( '@', user_name)) as domain 
# MAGIC  from project_users u, projects p 
# MAGIC  where p.project_id = u.project_id 
# MAGIC  and user_name not like '%apption.com') as a
# MAGIC  ) as b group by dept;
# MAGIC
# MAGIC  create or replace view project_workspace_spend as 
# MAGIC  select distinct Project_Acronym_CD, c.Current as spend 
# MAGIC  from project_credits c, projects p 
# MAGIC  where p.project_id = c.projectid;
