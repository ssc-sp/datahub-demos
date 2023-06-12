/* project_user_count */
select dept, count(9) as user_count from (
select case lower(substring(domain, 1, charindex('.', domain)-1)) when 'agr' then 'AAFC' when 'asc-csa'  then 'CSA' when 'cnrc-nrc' then 'NRC' when 'dfo-mpo' then 'DFO' when 'ec' then 'ECCC' when 'inspection' then 'CFIA' when 'nrcan' then 'NRCan' when 'nrcan-rncan' then 'NRCan' when 'nrc-cnrc' then 'NRC'when 'otc-cta' then 'CTA' when 'ssc-spc' then 'SSC' when 'hc-sc' then 'HC' else domain end as dept from (
SELECT RIGHT (user_name,
LEN(user_name) - CHARINDEX( '@', user_name)) AS Domain
FROM project_users) as emails where domain not like 'apption%') as departments group by dept;


/* project_workspace_count */
select dept, count(9) workspace_count from (
select Project_Acronym_CD, case lower(substring(domain, 1, charindex('.', domain)-1)) when 'agr' then 'AAFC' when 'asc-csa'  then 'CSA' when 'cnrc-nrc' then 'NRC' when 'dfo-mpo' then 'DFO' when 'ec' then 'ECCC' when 'inspection' then 'CFIA' when 'nrcan' then 'NRCan' when 'nrcan-rncan' then 'NRCan' when 'nrc-cnrc' then 'NRC'when 'otc-cta' then 'CTA' when 'ssc-spc' then 'SSC' when 'hc-sc' then 'HC' else domain end as dept
from
(select distinct Project_Acronym_CD, RIGHT (user_name,LEN(user_name) - CHARINDEX( '@', user_name)) as domain 
from project_users u, projects p 
where p.project_id = u.project_id 
and user_name not like '%apption.com') as a
) as b group by dept;

/*spend and budget*/
select distinct Project_Acronym_CD, c."Current", p.project_budget as budget 
from project_credits c, projects p 
where p.project_id = c.projectid;

/* project_user_department */
select distinct user_name, case lower(substring(domain, 1, charindex('.', domain)-1)) when 'agr' then 'AAFC' when 'asc-csa'  then 'CSA' when 'cnrc-nrc' then 'NRC' when 'dfo-mpo' then 'DFO' when 'ec' then 'ECCC' when 'inspection' then 'CFIA' when 'nrcan' then 'NRCan' when 'nrcan-rncan' then 'NRCan' when 'nrc-cnrc' then 'NRC'when 'otc-cta' then 'CTA' when 'ssc-spc' then 'SSC' when 'hc-sc' then 'HC' else domain end as dept from (
SELECT RIGHT (user_name,
LEN(user_name) - CHARINDEX( '@', user_name)) AS Domain, user_name, project_id
FROM project_users) as emails where domain not like 'apption%';

/*user with multiple projects*/
select user_name, count(9) from (
select  user_name, case lower(substring(domain, 1, charindex('.', domain)-1)) when 'agr' then 'AAFC' when 'asc-csa'  then 'CSA' when 'cnrc-nrc' then 'NRC' when 'dfo-mpo' then 'DFO' when 'ec' then 'ECCC' when 'inspection' then 'CFIA' when 'nrcan' then 'NRCan' when 'nrcan-rncan' then 'NRCan' when 'nrc-cnrc' then 'NRC'when 'otc-cta' then 'CTA' when 'ssc-spc' then 'SSC' when 'hc-sc' then 'HC' else domain end as dept from (
SELECT RIGHT (user_name,
LEN(user_name) - CHARINDEX( '@', user_name)) AS Domain, user_name, project_id
FROM project_users) as emails where domain not like 'apption%') as usercount group by user_name having count(9) >1;


/*Workspace with multiple SBDA*/
select sector_name, count(9) from (
select  distinct sector_name,case lower(substring(domain, 1, charindex('.', domain)-1)) when 'agr' then 'AAFC' when 'asc-csa'  then 'CSA' when 'cnrc-nrc' then 'NRC' when 'dfo-mpo' then 'DFO' when 'ec' then 'ECCC' when 'inspection' then 'CFIA' when 'nrcan' then 'NRCan' when 'nrcan-rncan' then 'NRCan' when 'nrc-cnrc' then 'NRC'when 'otc-cta' then 'CTA' when 'ssc-spc' then 'SSC' when 'hc-sc' then 'HC' else domain end as dept from (
SELECT RIGHT (user_name,
LEN(user_name) - CHARINDEX( '@', user_name)) AS Domain, user_name, p.sector_name
FROM project_users u, projects p where p.project_id = u.project_id) as emails where domain not like 'apption%') as wscount group by sector_name having count(9) >1;

