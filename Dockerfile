FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY data ./data
# Fill the leaderboard with the baseline models at build time
RUN cfdbench benchmark --reset > /dev/null

EXPOSE 8000
CMD ["cfdbench", "serve", "--host", "0.0.0.0", "--port", "8000"]
