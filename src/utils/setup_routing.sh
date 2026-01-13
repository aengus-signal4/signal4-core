#!/bin/bash

# === CONFIGURATION ===
ETH_SERVICE="Ethernet"
WIFI_SERVICE="Wi-Fi"
DEFAULT_NETMASK="255.255.255.0" # Assumes /24. Change if needed.

# === NODE IP CONFIGURATION ===
# Node IPs from config.yaml
# Format: (eth_ip wifi_ip node_name)
declare -a NODES=(
  "10.0.0.22 10.0.0.105 head"
  "10.0.0.129 10.0.0.66 worker1"
  "10.0.0.183 10.0.0.175 worker2"
  "10.0.0.203 10.0.0.252 worker3"
  "10.0.0.51 10.0.0.29 worker4"
  "10.0.0.209 10.0.0.97 worker5"
)

# === GET ETHERNET DEVICE NAME ===
ETH_DEVICE=$(networksetup -listallhardwareports | \
  awk -v service="$ETH_SERVICE" '
    $0 ~ "Hardware Port" && $3 == service {
      getline; if ($1 == "Device:") print $2
    }')

if [ -z "$ETH_DEVICE" ]; then
  echo "❌ Could not find device for $ETH_SERVICE"
  exit 1
fi

# === GET WIFI DEVICE NAME ===
WIFI_DEVICE=$(networksetup -listallhardwareports | \
  awk -v service="$WIFI_SERVICE" '
    $0 ~ "Hardware Port" && $3 == service {
      getline; if ($1 == "Device:") print $2
    }')

if [ -z "$WIFI_DEVICE" ]; then
  echo "❌ Could not find device for $WIFI_SERVICE"
  exit 1
fi

# === GET IP ADDRESSES ===
ETH_IP=$(ipconfig getifaddr "$ETH_DEVICE")
WIFI_IP=$(ipconfig getifaddr "$WIFI_DEVICE")

echo "📡 Ethernet interface: $ETH_DEVICE (IP: $ETH_IP)"
echo "📶 Wi-Fi interface: $WIFI_DEVICE (IP: $WIFI_IP)"

# === REMOVE EXISTING ROUTES FOR CLEANUP ===
echo "🧹 Cleaning up existing routes..."
# Save default route
DEFAULT_ROUTE=$(netstat -rn | grep default | head -1)

# Flush routing table (keeping default route)
sudo route -n flush

# Restore default route
if [ -n "$DEFAULT_ROUTE" ]; then
  DEFAULT_GW=$(echo "$DEFAULT_ROUTE" | awk '{print $2}')
  DEFAULT_INTERFACE=$(echo "$DEFAULT_ROUTE" | awk '{print $NF}')
  sudo route -n add default "$DEFAULT_GW"
fi

# === ADD ROUTES FOR EACH NODE ===
echo "🔄 Adding routes for cluster nodes..."

for node_info in "${NODES[@]}"; do
  read -r eth_ip wifi_ip node_name <<< "$node_info"
  
  # Primary route via Ethernet
  echo "  → $node_name: $eth_ip via Ethernet (primary)"
  sudo route -n add -host "$eth_ip" -interface "$ETH_DEVICE"
  
  # Fallback route via Wi-Fi
  echo "  → $node_name: $wifi_ip via Wi-Fi (fallback)"
  sudo route -n add -host "$wifi_ip" -interface "$WIFI_DEVICE"
done

# === CALCULATE SUBNET ===
if [ -n "$ETH_IP" ]; then
  IFS='.' read -r a b c d <<< "$ETH_IP"
  ETH_SUBNET="$a.$b.$c.0/24"
  echo "🧭 Routing Ethernet subnet: $ETH_SUBNET via Ethernet"
  sudo route -n add -net "$ETH_SUBNET" -interface "$ETH_DEVICE"
fi

# === SET SERVICE ORDER ===
ALL_SERVICES=$(networksetup -listallnetworkservices | tail -n +2 | sed 's/^\*//')

# Reorder so Wi-Fi is first for internet, but specific routes use Ethernet
echo "🔧 Prioritizing Wi-Fi for general internet access..."
networksetup -ordernetworkservices "$WIFI_SERVICE" "$ETH_SERVICE" $(
  echo "$ALL_SERVICES" | grep -v -e "$WIFI_SERVICE" -e "$ETH_SERVICE"
)

echo "✅ Network routing updated!"
echo "   - Ethernet is now default for all cluster nodes"
echo "   - Wi-Fi connections configured as fallback"
echo "   - Wi-Fi prioritized for general internet access"
echo "   Run 'netstat -rn' to verify routes."
