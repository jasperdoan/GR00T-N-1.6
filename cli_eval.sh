#!/bin/bash

# --- ANSI Color Codes ---
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# --- Configuration ---
PYTHON_EXEC="python"
SCRIPT_PATH="gr00t/eval/real_robot/SO100/eval_so100.py"
ROBOT_TYPE="so101_follower"
ROBOT_PORT="/dev/ttyACM0"
ROBOT_ID="follower_arm"
CAMERAS='{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 3, width: 640, height: 480, fps: 30}}'
POLICY_HOST="localhost"
POLICY_PORT="5555"
HORIZON=16

clear
echo -e "${CYAN}${BOLD}=============================================="
echo -e "      SO100 ROBOT EVALUATION INTERFACE        "
echo -e "==============================================${NC}"

while true; do
    echo -e "\n${BOLD}STEP 1: Select Task Type${NC}"
    echo -e "  1) ${GREEN}Check In${NC}  (Storage)"
    echo -e "  2) ${YELLOW}Check Out${NC} (Checkout Zone)"
    echo -e "  q) Quit"
    read -p "Selection [1-2, q]: " task_input

    case $task_input in
        1) TASK="Check in";;
        2) TASK="Check out";;
        q|Q) echo -e "${RED}Exiting...${NC}"; exit 0;;
        *) echo -e "${RED}Invalid selection.${NC}"; continue;;
    esac

    echo -e "\n${BOLD}STEP 2: Select Cube Color${NC}"
    echo -e "  1) ${RED}Red${NC}"
    echo -e "  2) ${CYAN}Blue${NC}"
    echo -e "  3) ${YELLOW}Yellow${NC}"
    echo -e "  b) Back to Task selection"
    read -p "Selection [1-3, b]: " color_input

    case $color_input in
        1) COLOR="red";;
        2) COLOR="blue";;
        3) COLOR="yellow";;
        b|B) continue;;
        *) echo -e "${RED}Invalid selection.${NC}"; continue;;
    esac

    # Construct full instruction
    LANG_INST="$TASK $COLOR cube"

    echo -e "\n${GREEN}${BOLD}>>> PREPARING EXECUTION <<<${NC}"
    echo -e "${BOLD}Instruction:${NC} $LANG_INST"
    echo -e "${BOLD}Robot Port :${NC} $ROBOT_PORT"
    echo -e "----------------------------------------------"

    # Run the command
    PYTHONPATH=. $PYTHON_EXEC $SCRIPT_PATH \
        --robot.type=$ROBOT_TYPE \
        --robot.port=$ROBOT_PORT \
        --robot.id=$ROBOT_ID \
        --robot.cameras="$CAMERAS" \
        --policy_host=$POLICY_HOST \
        --policy_port=$POLICY_PORT \
        --action_horizon=$HORIZON \
        --lang_instruction="$LANG_INST" \
        --robot.use_degrees=true

    echo -e "\n${CYAN}----------------------------------------------"
    echo -e "Task finished. Returning to menu..."
    echo -e "----------------------------------------------${NC}"
    sleep 1
done