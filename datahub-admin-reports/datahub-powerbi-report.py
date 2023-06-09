# Spark conf sample
# spark.databricks.cluster.profile singleNode
# spark.master local[*, 4]
# spark.readonly_username {{secrets/datahub/datahub-mssql-readonly-username}}
# spark.readonly_password {{secrets/datahub/datahub-mssql-readonly-password}}
# spark.databricks.delta.preview.enabled true
# spark.powerbi_readonly_password {{secrets/datahub/powerbi-readonly-password}}

# Databricks notebook source
spark.sql("drop table if exists project_users;")
spark.sql(
"CREATE TABLE project_users " +
"USING sqlserver OPTIONS ("+
  "dbtable 'dbo.project_users'," +
  "host 'fsdh-portal-sql-poc.database.windows.net',"+
  "port '1433',"+
  "database 'dh-portal-projectdb',"+
  "user 'powerbi_readonly',"+
  "password '"+ 
  spark.conf.get('spark.powerbi_readonly_password') + # CONFIG: spark.readonly_password {{secrets/datahub/datahub-mssql-readonly-password}}
  "');")
  

# COMMAND ----------

spark.sql("drop table if exists project_costs;")
spark.sql(
"CREATE TABLE project_costs " +
"USING sqlserver OPTIONS ("+
  "dbtable 'dbo.project_costs'," +
  "host 'fsdh-portal-sql-poc.database.windows.net',"+
  "port '1433',"+
  "database 'dh-portal-projectdb',"+
  "user 'powerbi_readonly',"+
  "password '"+ 
  spark.conf.get('spark.powerbi_readonly_password') + # CONFIG: spark.readonly_password {{secrets/datahub/datahub-mssql-readonly-password}}
  "');")
  

# COMMAND ----------

spark.sql("drop table if exists project_credits;")
spark.sql(
"CREATE TABLE project_credits " +
"USING sqlserver OPTIONS ("+
  "dbtable 'dbo.project_credits'," +
  "host 'fsdh-portal-sql-poc.database.windows.net',"+
  "port '1433',"+
  "database 'dh-portal-projectdb',"+
  "user 'powerbi_readonly',"+
  "password '"+ 
  spark.conf.get('spark.powerbi_readonly_password') + # CONFIG: spark.readonly_password {{secrets/datahub/datahub-mssql-readonly-password}}
  "');")
  

# COMMAND ----------

spark.sql("drop table if exists projects;")
spark.sql(
"CREATE TABLE projects " +
"USING sqlserver OPTIONS ("+
  "dbtable 'dbo.projects'," +
  "host 'fsdh-portal-sql-poc.database.windows.net',"+
  "port '1433',"+
  "database 'dh-portal-projectdb',"+
  "user 'powerbi_readonly',"+
  "password '"+ 
  spark.conf.get('spark.powerbi_readonly_password') + # CONFIG: spark.readonly_password {{secrets/datahub/datahub-mssql-readonly-password}}
  "');")
  

# COMMAND ----------

 %sql

 create or replace view project_user_count as
 select dept, count(9) as user_count from (
 select case lower(substring(domain, 1, charindex('.', domain)-1)) when 'agr' then 'AAFC' when 'asc-csa'  then 'CSA' when 'cnrc-nrc' then 'NRC' when 'dfo-mpo' then 'DFO' when 'ec' then 'ECCC' when 'inspection' then 'CFIA' when 'nrcan' then 'NRCan' when 'nrcan-rncan' then 'NRCan' when 'nrc-cnrc' then 'NRC'when 'otc-cta' then 'CTA' when 'ssc-spc' then 'SSC' else domain end as dept from (
 SELECT RIGHT (user_name,
 LEN(user_name) - CHARINDEX( '@', user_name)) AS Domain
 FROM project_users) as emails where domain not like 'apption%') as departments group by dept

# COMMAND ----------

 %sql

 create or replace view project_workspace_count as 
 select dept, count(9) workspace_count from (
 select Project_Acronym_CD, case lower(substring(domain, 1, charindex('.', domain)-1)) when 'agr' then 'AAFC' when 'asc-csa'  then 'CSA' when 'cnrc-nrc' then 'NRC' when 'dfo-mpo' then 'DFO' when 'ec' then 'ECCC' when 'inspection' then 'CFIA' when 'nrcan' then 'NRCan' when 'nrcan-rncan' then 'NRCan' when 'nrc-cnrc' then 'NRC'when 'otc-cta' then 'CTA' when 'ssc-spc' then 'SSC' else domain end as dept
 from
 (select distinct Project_Acronym_CD, RIGHT (user_name,LEN(user_name) - CHARINDEX( '@', user_name)) as domain 
 from project_users u, projects p 
 where p.project_id = u.project_id 
 and user_name not like '%apption.com') as a
 ) as b group by dept

# COMMAND ----------

 %sql

 create or replace view project_workspace_spend as 
 select distinct Project_Acronym_CD, c.Current as spend 
 from project_credits c, projects p 
 where p.project_id = c.projectid
