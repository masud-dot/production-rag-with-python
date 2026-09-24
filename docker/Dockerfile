# Build fat, ship thin. Chapter 26.
# NOT verified by build in the authoring environment — see Appendix A.
FROM python:3.13-slim AS build
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.13-slim
COPY --from=build /install /usr/local
COPY src/ /app/src/
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "lighthouse.api.routes:app", "--host", "0.0.0.0"]
