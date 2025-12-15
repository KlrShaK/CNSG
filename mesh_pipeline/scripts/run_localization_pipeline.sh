#!/bin/bash
# Run single image localization pipeline with optional visualization
#
# Usage:
#   ./run_localization_pipeline.sh --query-image /path/to/image.jpg
#   ./run_localization_pipeline.sh --query-image /path/to/image.jpg --visualize
#   ./run_localization_pipeline.sh --query-image /path/to/image.jpg --output-dir ./my_outputs --visualize

set -e  # Exit on error

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default values
QUERY_IMAGE=""
OUTPUT_DIR=""
VISUALIZE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --query-image)
            QUERY_IMAGE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --visualize)
            VISUALIZE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --query-image <path> [--output-dir <path>] [--visualize] [--pose-format c2w|w2c]"
            echo ""
            echo "Arguments:"
            echo "  --query-image    Path to the query image to localize (required)"
            echo "  --output-dir     Output directory for results (optional, uses config default)"
            echo "  --visualize      Visualize the localized pose on the mesh (optional)"
            echo ""
            echo "Examples:"
            echo "  $0 --query-image ./test.jpg"
            echo "  $0 --query-image ./test.jpg --visualize"
            echo "  $0 --query-image ./test.jpg --output-dir ./my_outputs --visualize"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "${QUERY_IMAGE}" ]]; then
    echo "Error: --query-image is required"
    echo "Use --help for usage information"
    exit 1
fi

if [[ ! -f "${QUERY_IMAGE}" ]]; then
    echo "Error: Query image not found: ${QUERY_IMAGE}"
    exit 1
fi

# Convert to absolute path
QUERY_IMAGE="$(cd "$(dirname "${QUERY_IMAGE}")" && pwd)/$(basename "${QUERY_IMAGE}")"

echo "=========================================="
echo "Running Localization Pipeline"
echo "=========================================="
echo "Query image: ${QUERY_IMAGE}"
echo "Project root: ${PROJECT_ROOT}"
echo ""

# Step 1: Run localization
echo "Step 1/2: Running localization..."
echo ""

cd "${PROJECT_ROOT}/mesh_pipeline/src/localization"

if [[ -z "${OUTPUT_DIR}" ]]; then
    python3 run_localization.py --query_image "${QUERY_IMAGE}"
else
    python3 run_localization.py --query_image "${QUERY_IMAGE}" --output_dir "${OUTPUT_DIR}"
fi

# Check if localization succeeded
if [[ $? -ne 0 ]]; then
    echo ""
    echo "Error: Localization failed!"
    exit 1
fi

echo ""
echo "✓ Localization complete!"

# Step 2: Visualize if requested
if [[ "${VISUALIZE}" == true ]]; then
    echo ""
    echo "Step 2/2: Visualizing pose..."
    echo ""

    # Construct path to poses.txt
    # LaMAR saves to: outputs/pose_estimation/query_single/map/superpoint/superglue/netvlad-10/triangulation/single_image/poses.txt
    LAMAR_REPO="${PROJECT_ROOT}/mesh_pipeline/third_party/lamar-benchmark"
    POSES_FILE="${PROJECT_ROOT}/mesh_pipeline/outputs/pose_estimation/query_single/map/superpoint/superglue/netvlad-10/triangulation/single_image/poses.txt"

    if [[ ! -f "${POSES_FILE}" ]]; then
        echo "Warning: Poses file not found at expected location:"
        echo "  ${POSES_FILE}"
        echo ""
        echo "Searching for poses.txt in outputs directory..."

        # Try to find poses.txt in the outputs
        FOUND_POSES=$(find "${PROJECT_ROOT}/mesh_pipeline/outputs" -name "poses.txt" -type f 2>/dev/null | head -1)

        if [[ -z "${FOUND_POSES}" ]]; then
            echo "Error: Could not find poses.txt in outputs directory"
            exit 1
        fi

        POSES_FILE="${FOUND_POSES}"
        echo "Found poses at: ${POSES_FILE}"
    fi

    echo "Visualizing pose from: ${POSES_FILE}"
    echo "Pose format: ${POSE_FORMAT}"
    echo ""

    # Run visualization
    python3 visualize_pose.py \
        --poses "${POSES_FILE}" \
        --apply-mesh-alignment

    if [[ $? -ne 0 ]]; then
        echo ""
        echo "Error: Visualization failed!"
        exit 1
    fi

    echo ""
    echo "✓ Visualization complete!"
fi

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
