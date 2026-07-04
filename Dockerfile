FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-broken \
        ffmpeg \
        poppler-utils \
        libmagic1 \
        # Build tools (needed for pycairo / manim)
        build-essential \
        python3-dev \
        meson \
        ninja-build \
        # Manim rendering dependencies
        libcairo2-dev \
        libpango1.0-dev \
        texlive-latex-base \
        texlive-latex-extra \
        texlive-fonts-recommended \
        texlive-science \
        dvisvgm \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run DB migration then start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
