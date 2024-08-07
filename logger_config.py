import logging

# Configure logging
logging.basicConfig(
    level=logging.WARN,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("simulation.log"), logging.StreamHandler()],
)

logger = logging.getLogger("logger")
