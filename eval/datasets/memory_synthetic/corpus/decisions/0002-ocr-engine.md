# 0002 — Keep the local OCR engine

Date: 2026-10-15. Status: accepted.

We keep the local OCR engine instead of the cloud OCR API. On our 300-page test set the cloud API was 3 points more accurate on handwriting, but it costs EUR 0.012 per page and sends client documents off-site, which two prospects refused. Revisit when a client asks for handwriting.
