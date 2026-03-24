#!/usr/bin/env python3
"""Batch generation script for testing production generation endpoints."""

import argparse
import asyncio
import signal
import statistics
from pathlib import Path
from urllib.parse import urlparse

import httpx
from loguru import logger
from pydantic import BaseModel


# Default paths (can be overridden via command-line arguments)
DEFAULT_INPUT = "./input_images"
# DEFAULT_ENDPOINT = "http://213.181.104.57:16263"
DEFAULT_ENDPOINT = "http://localhost:10006"

DEFAULT_OUTPUT_DIR = None  # If None, defaults to <input_dir>_output_seed_<seed> or <path_parts>_seed_<seed> for URL
DEFAULT_SEED = 42

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

DEFAULT_GENERATION_TIMEOUT = 300.0
DEFAULT_DOWNLOAD_TIMEOUT = 120.0
DEFAULT_CONCURRENCY = 1

class GracefulShutdown:
    """Graceful shutdown handler."""

    def __init__(self) -> None:
        self.should_stop = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: object) -> None:
        logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.should_stop = True


class GenerationResponse(BaseModel):
    """Response from a generation request."""

    success: bool = False
    content: bytes | None = None
    generation_time: float | None = None
    download_time: float | None = None


async def generate(
    request_sem: asyncio.Semaphore,
    endpoint: str,
    image: bytes,
    seed: int,
    log_id: str,
    shutdown: GracefulShutdown,
    timeout: httpx.Timeout,
) -> GenerationResponse:
    """Generate a 3D model from image bytes."""
    if shutdown.should_stop:
        logger.debug(f"{log_id}: cancelled, shutdown in progress")
        return GenerationResponse(success=False)

    sem_released = False
    await request_sem.acquire()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                start_time = asyncio.get_running_loop().time()

                logger.debug(f"{log_id}: generating")

                async with client.stream(
                    "POST",
                    f"{endpoint}/generate",
                    files={"prompt_image_file": ("prompt.jpg", image, "image/jpeg")},
                    data={"seed": seed},
                ) as response:
                    response.raise_for_status()

                    elapsed = asyncio.get_running_loop().time() - start_time

                    # Release the slot as soon as generation finishes (first bytes).
                    request_sem.release()
                    sem_released = True

                    logger.debug(f"{log_id}: generation completed in {elapsed:.1f}s")

                    try:
                        content = await response.aread()
                    except Exception as e:
                        logger.warning(f"{log_id}: failed to read response body: {e}")
                        return GenerationResponse(success=False)

                    download_time = asyncio.get_running_loop().time() - start_time - elapsed
                    mb_size = len(content) / 1024 / 1024
                    logger.debug(
                        f"Generated for {log_id} in {elapsed:.1f}s, "
                        f"downloaded in {download_time:.1f}s, {mb_size:.1f} MiB"
                    )

                    return GenerationResponse(
                        success=True,
                        content=content,
                        generation_time=elapsed,
                        download_time=download_time,
                    )

            except httpx.TimeoutException:
                logger.warning(f"{log_id}: generation timed out")
                return GenerationResponse(success=False)

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                try:
                    error_body = await exc.response.aread()
                    error_preview = error_body.decode("utf-8", errors="replace")[:500]
                except Exception:
                    error_preview = "(could not read body)"

                logger.error(f"{log_id}: HTTP {status}: {error_preview}")
                return GenerationResponse(success=False)

            except httpx.RequestError as e:
                # Connection errors, DNS failures, etc.
                logger.warning(f"{log_id}: request error: {e}")
                return GenerationResponse(success=False)

            except Exception as e:
                logger.error(f"{log_id}: unexpected error during generation: {e}")
                return GenerationResponse(success=False)

    finally:
        if not sem_released:
            request_sem.release()


def iter_images(input_dir: Path) -> list[Path]:
    """Get sorted list of image files from directory."""
    return sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def is_url(path: str) -> bool:
    """Check if the given path is a URL."""
    parsed = urlparse(path)
    return parsed.scheme in ("http", "https")


def _github_blob_to_raw(url: str) -> str:
    """Convert a GitHub blob URL to a raw content URL."""
    parsed = urlparse(url)
    if parsed.netloc == "github.com" and "/blob/" in parsed.path:
        # Replace host and remove '/blob' segment from the path
        new_path = parsed.path.replace("/blob/", "/", 1)
        return f"https://raw.githubusercontent.com{new_path}"
    return url


async def fetch_seed_from_url(base_url: str) -> int | None:
    """Fetch seed from seed.json at the same parent directory as the input URL."""
    # Get parent directory URL and append seed.json
    base_url = _github_blob_to_raw(base_url)
    parsed = urlparse(base_url)
    parent_path = "/".join(parsed.path.rsplit("/", 1)[:-1]) + "/"
    seed_url = f"{parsed.scheme}://{parsed.netloc}{parent_path}seed.json"

    logger.debug(f"Attempting to fetch seed from: {seed_url}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(seed_url)
            response.raise_for_status()
            data = response.json()
            seed = data.get("seed")
            if seed is not None:
                logger.info(f"Fetched seed from {seed_url}: {seed}")
                return int(seed)
            else:
                logger.warning(f"seed.json found but no 'seed' key: {data}")
                return None
    except httpx.HTTPStatusError as e:
        logger.warning(f"Failed to fetch seed.json (HTTP {e.response.status_code}): {seed_url}")
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch seed.json: {e}")
        return None


async def fetch_image_urls_from_url(url: str) -> list[str]:
    """Fetch list of image URLs from a text file URL."""
    url = _github_blob_to_raw(url)
    logger.info(f"Fetching image list from: {url}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        content = response.text

    # Parse lines, strip whitespace, filter empty lines and comments
    lines = [line.strip() for line in content.splitlines()]
    image_urls = [line for line in lines if line and not line.startswith("#")]

    logger.info(f"Found {len(image_urls)} image URLs")
    return image_urls


async def download_image(url: str, client: httpx.AsyncClient) -> tuple[str, bytes | None]:
    """Download an image from a URL."""
    try:
        response = await client.get(url)
        response.raise_for_status()
        return url, response.content
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        return url, None


def write_timing_summary(
    summary_path: Path,
    images: list[Path],
    generation_times: list[float],
    download_times: list[float],
    total_times: list[float],
    output_sizes: list[int],
) -> None:
    """Write timing summary to file."""
    successful_indices = [i for i, t in enumerate(total_times) if t > 0]

    if not successful_indices:
        logger.warning("No successful generations to summarize")
        return

    successful_gen_times = [generation_times[i] for i in successful_indices]
    successful_dl_times = [download_times[i] for i in successful_indices]
    successful_total_times = [total_times[i] for i in successful_indices]
    successful_sizes = [output_sizes[i] for i in successful_indices]

    size_divisor = 1024 * 1024

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("=" * 60 + "\n")
        handle.write("GENERATION TIMING SUMMARY\n")
        handle.write("=" * 60 + "\n\n")

        handle.write(f"Total images processed: {len(images)}\n")
        handle.write(f"Successful generations: {len(successful_indices)}\n")
        handle.write(f"Failed generations: {len(images) - len(successful_indices)}\n\n")

        handle.write("-" * 40 + "\n")
        handle.write("Generation time stats (seconds)\n")
        handle.write("-" * 40 + "\n")
        handle.write(f"  Average: {statistics.mean(successful_gen_times):.4f}\n")
        handle.write(f"  Max:     {max(successful_gen_times):.4f}\n")
        handle.write(f"  Min:     {min(successful_gen_times):.4f}\n")
        if len(successful_gen_times) > 1:
            handle.write(f"  Std Dev: {statistics.stdev(successful_gen_times):.4f}\n")

        handle.write("\n")
        handle.write("-" * 40 + "\n")
        handle.write("Download time stats (seconds)\n")
        handle.write("-" * 40 + "\n")
        handle.write(f"  Average: {statistics.mean(successful_dl_times):.4f}\n")
        handle.write(f"  Max:     {max(successful_dl_times):.4f}\n")
        handle.write(f"  Min:     {min(successful_dl_times):.4f}\n")
        if len(successful_dl_times) > 1:
            handle.write(f"  Std Dev: {statistics.stdev(successful_dl_times):.4f}\n")

        handle.write("\n")
        handle.write("-" * 40 + "\n")
        handle.write("Total time stats (seconds)\n")
        handle.write("-" * 40 + "\n")
        handle.write(f"  Average: {statistics.mean(successful_total_times):.4f}\n")
        handle.write(f"  Max:     {max(successful_total_times):.4f}\n")
        handle.write(f"  Min:     {min(successful_total_times):.4f}\n")
        if len(successful_total_times) > 1:
            handle.write(f"  Std Dev: {statistics.stdev(successful_total_times):.4f}\n")

        handle.write("\n")
        handle.write("-" * 40 + "\n")
        handle.write("File size stats (MB)\n")
        handle.write("-" * 40 + "\n")
        avg_size = statistics.mean(successful_sizes) / size_divisor
        max_size = max(successful_sizes) / size_divisor
        min_size = min(successful_sizes) / size_divisor
        handle.write(f"  Average: {avg_size:.2f}\n")
        handle.write(f"  Max:     {max_size:.2f}\n")
        handle.write(f"  Min:     {min_size:.2f}\n")

        handle.write("\n")
        handle.write("=" * 60 + "\n")
        handle.write("DETAILED TIMING\n")
        handle.write("=" * 60 + "\n")
        handle.write(f"{'Filename':<40} {'Gen(s)':>10} {'DL(s)':>10} {'Total(s)':>10} {'Size(MB)':>10}\n")
        handle.write("-" * 80 + "\n")

        for image_path, gen_t, dl_t, total_t, size in zip(
            images, generation_times, download_times, total_times, output_sizes
        ):
            status = "✓" if total_t > 0 else "✗"
            handle.write(
                f"{status} {image_path.name:<38} {gen_t:>10.4f} {dl_t:>10.4f} "
                f"{total_t:>10.4f} {size / size_divisor:>10.2f}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch generation script for testing production endpoints. "
            "Sends images from a folder or URL to the generation endpoint and saves resulting GLB files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i", "--input",
        default=DEFAULT_INPUT,
        help="Input source: local folder with images OR URL to text file with image URLs.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder for generated models. Defaults to <input_dir>_output_seed_<seed> for local, <path>_seed_<seed> for URL.",
    )
    parser.add_argument(
        "-e", "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Generation endpoint base URL (without /generate).",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help=f"Seed to pass to the server. Default: {DEFAULT_SEED}. For URL input, auto-fetched from seed.json if not specified.",
    )
    parser.add_argument(
        "-c", "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Maximum concurrent requests.",
    )
    parser.add_argument(
        "--generation-timeout",
        type=float,
        default=DEFAULT_GENERATION_TIMEOUT,
        help="Generation timeout in seconds.",
    )
    parser.add_argument(
        "--download-timeout",
        type=float,
        default=DEFAULT_DOWNLOAD_TIMEOUT,
        help="Download timeout in seconds.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of images to process.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging.",
    )

    return parser.parse_args()


async def process_single_image(
    image_path: Path,
    output_dir: Path,
    request_sem: asyncio.Semaphore,
    endpoint: str,
    seed: int,
    shutdown: GracefulShutdown,
    timeout: httpx.Timeout,
) -> tuple[Path, GenerationResponse]:
    """Process a single image from local path and return result."""
    log_id = image_path.name

    logger.info(f"Processing {log_id}...")

    image_data = image_path.read_bytes()

    result = await generate(
        request_sem=request_sem,
        endpoint=endpoint,
        image=image_data,
        seed=seed,
        log_id=log_id,
        shutdown=shutdown,
        timeout=timeout,
    )

    return image_path, result


async def process_single_image_from_bytes(
    image_name: str,
    image_data: bytes,
    request_sem: asyncio.Semaphore,
    endpoint: str,
    seed: int,
    shutdown: GracefulShutdown,
    timeout: httpx.Timeout,
) -> tuple[str, GenerationResponse]:
    """Process a single image from bytes and return result."""
    logger.info(f"Processing {image_name}...")

    result = await generate(
        request_sem=request_sem,
        endpoint=endpoint,
        image=image_data,
        seed=seed,
        log_id=image_name,
        shutdown=shutdown,
        timeout=timeout,
    )

    return image_name, result


def _get_image_name_from_url(url: str) -> str:
    """Extract image filename from URL."""
    parsed = urlparse(url)
    path = parsed.path
    name = path.rsplit("/", 1)[-1] if "/" in path else path
    # Remove query string if present
    if "?" in name:
        name = name.split("?")[0]
    return name or "image"


async def async_main() -> int:
    args = parse_args()

    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(
            lambda msg: print(msg, end=""),
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
            level="DEBUG",
        )
    else:
        logger.remove()
        logger.add(
            lambda msg: print(msg, end=""),
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
            level="INFO",
        )

    # Determine if input is URL or local directory
    input_source = args.input
    is_url_input = is_url(input_source)

    # Determine seed
    if args.seed is not None:
        # User explicitly provided seed
        seed = args.seed
        logger.info(f"Using provided seed: {seed}")
    elif is_url_input:
        # Try to fetch seed from seed.json
        fetched_seed = await fetch_seed_from_url(input_source)
        if fetched_seed is not None:
            seed = fetched_seed
        else:
            seed = DEFAULT_SEED
            logger.info(f"Using default seed: {seed}")
    else:
        seed = DEFAULT_SEED
        logger.info(f"Using default seed: {seed}")

    if is_url_input:
        # URL mode: fetch image URLs and process them
        logger.info(f"URL input mode: {input_source}")

        try:
            image_urls = await fetch_image_urls_from_url(input_source)
        except Exception as e:
            logger.error(f"Failed to fetch image list from URL: {e}")
            return 1

        if args.limit:
            image_urls = image_urls[: args.limit]

        if not image_urls:
            logger.info("No image URLs found")
            return 0

        # Setup output directory
        if args.output_dir:
            output_dir = Path(args.output_dir).expanduser().resolve()
        else:
            # Use a default output directory based on URL path
            # e.g., https://github.com/.../rounds/1/prompts.txt -> rounds_1_prompts_seed_XXX
            parsed_url = urlparse(input_source)
            path_parts = [p for p in parsed_url.path.split("/") if p]
            # Get the meaningful parts (skip repo structure like 'blob/main')
            # Find index of 'rounds' or similar meaningful start, or use last 3 parts
            meaningful_parts = []
            for i, part in enumerate(path_parts):
                if part in ("rounds",) or (meaningful_parts and len(meaningful_parts) < 3):
                    meaningful_parts.append(part)
                elif part not in ("blob", "main", "master", "raw"):
                    # Keep collecting if we haven't started yet
                    continue
            # Fallback: use last few path components
            if not meaningful_parts:
                meaningful_parts = path_parts[-3:] if len(path_parts) >= 3 else path_parts
            # Remove file extension from last part
            if meaningful_parts:
                meaningful_parts[-1] = Path(meaningful_parts[-1]).stem
            url_name = "_".join(meaningful_parts) or "url_input"
            output_dir = Path.cwd() / f"{url_name}_seed_{seed}"

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Get image names and filter already processed
        image_items: list[tuple[str, str]] = []  # (url, name)
        for url in image_urls:
            name = _get_image_name_from_url(url)
            stem = Path(name).stem
            if not (output_dir / f"{stem}.glb").exists():
                image_items.append((url, name))

        skipped = len(image_urls) - len(image_items)
        if skipped > 0:
            logger.info(f"Skipping {skipped} images with existing outputs")

        if not image_items:
            logger.info("All images already processed")
            return 0

        logger.info(f"Processing {len(image_items)} images")

        # Initialize
        shutdown = GracefulShutdown()
        request_sem = asyncio.Semaphore(args.concurrency)

        timeout = httpx.Timeout(
            connect=30.0,
            write=30.0,
            read=args.generation_timeout + args.download_timeout,
            pool=30.0,
        )

        # Track results
        failures: list[str] = []
        image_names: list[str] = []
        generation_times: list[float] = []
        download_times: list[float] = []
        total_times: list[float] = []
        output_sizes: list[int] = []

        # Process images
        async with httpx.AsyncClient(timeout=60.0) as download_client:
            for url, name in image_items:
                if shutdown.should_stop:
                    logger.warning("Shutdown requested, stopping processing...")
                    break

                stem = Path(name).stem
                output_path = output_dir / f"{stem}.glb"
                image_names.append(name)

                # Download image
                _, image_data = await download_image(url, download_client)
                if image_data is None:
                    failures.append(name)
                    output_sizes.append(0)
                    generation_times.append(0.0)
                    download_times.append(0.0)
                    total_times.append(0.0)
                    logger.error(f"✗ Failed to download {name}")
                    continue

                # Process image
                _, result = await process_single_image_from_bytes(
                    image_name=name,
                    image_data=image_data,
                    request_sem=request_sem,
                    endpoint=args.endpoint,
                    seed=seed,
                    shutdown=shutdown,
                    timeout=timeout,
                )

                if result.success and result.content:
                    output_path.write_bytes(result.content)
                    output_sizes.append(len(result.content))

                    gen_time = result.generation_time or 0.0
                    dl_time = result.download_time or 0.0
                    generation_times.append(gen_time)
                    download_times.append(dl_time)
                    total_times.append(gen_time + dl_time)

                    logger.info(
                        f"✓ Saved {output_path.name} ({len(result.content) / 1024 / 1024:.2f} MB) "
                        f"- gen: {gen_time:.2f}s, download: {dl_time:.2f}s"
                    )
                else:
                    failures.append(name)
                    output_sizes.append(0)
                    generation_times.append(0.0)
                    download_times.append(0.0)
                    total_times.append(0.0)
                    logger.error(f"✗ Failed to generate for {name}")

        # Write timing summary (create Path objects for compatibility)
        summary_path = output_dir / "timing_summary.txt"
        image_paths = [Path(name) for name in image_names]
        write_timing_summary(
            summary_path=summary_path,
            images=image_paths,
            generation_times=generation_times,
            download_times=download_times,
            total_times=total_times,
            output_sizes=output_sizes,
        )

        # Final summary
        successful = len(image_names) - len(failures)
        logger.info("=" * 50)
        logger.info(f"Batch processing complete: {successful}/{len(image_names)} successful")

        if failures:
            failed_names = ", ".join(failures)
            logger.error(f"Failed images: {failed_names}")

        if successful > 0:
            avg_gen = statistics.mean([t for t in generation_times if t > 0])
            avg_dl = statistics.mean([t for t in download_times if t > 0])
            logger.info(f"Average generation time: {avg_gen:.2f}s")
            logger.info(f"Average download time: {avg_dl:.2f}s")

        logger.info(f"Timing summary saved to: {summary_path}")

        return 1 if failures else 0

    else:
        # Local directory mode
        input_dir = Path(input_source).expanduser().resolve()

        if not input_dir.exists() or not input_dir.is_dir():
            logger.error(f"Input directory not found: {input_dir}")
            return 1

        if args.output_dir:
            output_dir = Path(args.output_dir).expanduser().resolve()
        else:
            output_dir = input_dir.parent / f"{input_dir.name}_output_seed_{seed}"

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")

        # Get images
        images = iter_images(input_dir)
        if args.limit:
            images = images[: args.limit]

        if not images:
            logger.info(f"No images found in {input_dir}")
            return 0

        logger.info(f"Found {len(images)} images to process")

        # Skip images with existing outputs
        original_count = len(images)
        images = [img for img in images if not (output_dir / f"{img.stem}.glb").exists()]
        skipped = original_count - len(images)
        if skipped > 0:
            logger.info(f"Skipping {skipped} images with existing outputs")

        if not images:
            logger.info("All images already processed")
            return 0

        # Initialize
        shutdown = GracefulShutdown()
        request_sem = asyncio.Semaphore(args.concurrency)

        timeout = httpx.Timeout(
            connect=30.0,
            write=30.0,
            read=args.generation_timeout + args.download_timeout,
            pool=30.0,
        )

        # Track results
        failures: list[Path] = []
        generation_times: list[float] = []
        download_times: list[float] = []
        total_times: list[float] = []
        output_sizes: list[int] = []

        # Process images
        for image_path in images:
            if shutdown.should_stop:
                logger.warning("Shutdown requested, stopping processing...")
                break

            output_path = output_dir / f"{image_path.stem}.glb"

            image_path, result = await process_single_image(
                image_path=image_path,
                output_dir=output_dir,
                request_sem=request_sem,
                endpoint=args.endpoint,
                seed=seed,
                shutdown=shutdown,
                timeout=timeout,
            )

            if result.success and result.content:
                output_path.write_bytes(result.content)
                output_sizes.append(len(result.content))

                gen_time = result.generation_time or 0.0
                dl_time = result.download_time or 0.0
                generation_times.append(gen_time)
                download_times.append(dl_time)
                total_times.append(gen_time + dl_time)

                logger.info(
                    f"✓ Saved {output_path.name} ({len(result.content) / 1024 / 1024:.2f} MB) "
                    f"- gen: {gen_time:.2f}s, download: {dl_time:.2f}s"
                )
            else:
                failures.append(image_path)
                output_sizes.append(0)
                generation_times.append(0.0)
                download_times.append(0.0)
                total_times.append(0.0)
                logger.error(f"✗ Failed to generate for {image_path.name}")

        # Write timing summary
        summary_path = output_dir / "timing_summary.txt"
        write_timing_summary(
            summary_path=summary_path,
            images=images,
            generation_times=generation_times,
            download_times=download_times,
            total_times=total_times,
            output_sizes=output_sizes,
        )

        # Final summary
        successful = len(images) - len(failures)
        logger.info("=" * 50)
        logger.info(f"Batch processing complete: {successful}/{len(images)} successful")

        if failures:
            failed_names = ", ".join(path.name for path in failures)
            logger.error(f"Failed images: {failed_names}")

        if successful > 0:
            avg_gen = statistics.mean([t for t in generation_times if t > 0])
            avg_dl = statistics.mean([t for t in download_times if t > 0])
            logger.info(f"Average generation time: {avg_gen:.2f}s")
            logger.info(f"Average download time: {avg_dl:.2f}s")

        logger.info(f"Timing summary saved to: {summary_path}")

        return 1 if failures else 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
