FROM postgres:17-alpine
COPY staybuddy_schema.sql /docker-entrypoint-initdb.d/000_base.sql
COPY migrations/*.sql /docker-entrypoint-initdb.d/
