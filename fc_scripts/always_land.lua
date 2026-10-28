-- always_land.lua (/APM/scripts): lands the drone on the FC itself, independent of the laptop and radio.
--
-- While armed and flying, switch to LAND when:
--   TOO HIGH       laser above ALAND_MAX_HGT for 0.5 s
--   LASER LOST     no good laser reading for 1 s
--   TILTED         roll or pitch beyond ALAND_TILT for 0.5 s
--   NOT FOLLOWING  attitude error beyond ALAND_ATT_ERR for 0.5 s
--   CAMERA BLIND   floor camera quality below 50 for 2 s in LOITER
-- TILTED, NOT FOLLOWING and CAMERA BLIND apply only 10 cm or more above the floor.
-- LAND rising 0.4 m above its lowest point for 1 s stops the motors.
-- ALAND_TEST = 1: also checks while disarmed and only reports, never acts.

local PARAM_TABLE_KEY = 74
local PREFIX = "ALAND_"
local MODE_LOITER, MODE_LAND = 5, 9
local DOWN = 25                       -- rangefinder orientation: pointing down
local RANGEFINDER_GOOD = 4
local FLOW_MIN_QUALITY = 50
local LAND_CLIMB_M, LAND_CLIMB_S = 0.4, 1.0
local AIRBORNE_M = 0.10                -- tilt/follow/camera rules start here

assert(param:add_table(PARAM_TABLE_KEY, PREFIX, 8), "always_land: could not add param table")
local function bind(index, name, default)
  assert(param:add_param(PARAM_TABLE_KEY, index, name, default), "always_land: could not add " .. name)
  return Parameter(PREFIX .. name)
end
local ENABLE  = bind(1, "ENABLE", 1)
local MAX_HGT = bind(2, "MAX_HGT", 2.3)   -- metres above the floor (laser)
local TILT    = bind(3, "TILT", 25)       -- degrees
local ATT_ERR = bind(4, "ATT_ERR", 6)     -- degrees
local TEST    = bind(5, "TEST", 0)

local since = {}                          -- rule name -> millis() when it first became true
local land_low, last_report = nil, nil

local function held(name, condition, seconds, now)
  if not condition then
    since[name] = nil
    return false
  end
  since[name] = since[name] or now
  return (now - since[name]) >= seconds * 1000
end

local function laser_height()
  if rangefinder:has_data_orient(DOWN) and rangefinder:status_orient(DOWN) == RANGEFINDER_GOOD then
    return rangefinder:distance_orient(DOWN)
  end
  return nil
end

local function attitude_error_deg()
  if AC_AttitudeControl == nil then
    return nil
  end
  local ok, value = pcall(function() return AC_AttitudeControl:get_att_error_angle_deg() end)
  if ok then
    return value
  end
  return nil
end

-- the first rule that fires, or nil
local function reason_to_land(now, mode)
  local height = laser_height()
  if held("high", height ~= nil and height > MAX_HGT:get(), 0.5, now) then
    return string.format("too high %.2f m", height)
  end
  if held("laser", height == nil, 1.0, now) then
    return "laser lost"
  end
  local airborne = height ~= nil and height > AIRBORNE_M
  local tilt = math.deg(math.max(math.abs(ahrs:get_roll_rad()), math.abs(ahrs:get_pitch_rad())))
  if held("tilt", airborne and tilt > TILT:get(), 0.5, now) then
    return string.format("tilted %.0f deg", tilt)
  end
  local err = attitude_error_deg()
  if held("follow", airborne and err ~= nil and err > ATT_ERR:get(), 0.5, now) then
    return string.format("not following, %.1f deg off", err)
  end
  if held("blind", airborne and mode == MODE_LOITER and optical_flow:quality() < FLOW_MIN_QUALITY, 2.0, now) then
    return "floor camera blind"
  end
  return nil
end

-- LAND must go down; true when it has been rising for too long
local function land_is_climbing(now)
  local height = laser_height()
  if height == nil then
    return false
  end
  land_low = math.min(land_low or height, height)
  return held("land_climb", height > land_low + LAND_CLIMB_M, LAND_CLIMB_S, now)
end

local function report(text, now)
  if last_report == nil or now - last_report >= 2000 then
    gcs:send_text(4, "ALWAYS-LAND " .. text)
    last_report = now
  end
end

local function update()
  local now = millis():tofloat()
  if ENABLE:get() < 1 then
    return
  end
  local testing = TEST:get() >= 1
  local flying = arming:is_armed() and vehicle:get_likely_flying()
  if not flying and not testing then
    since, land_low = {}, nil
    return
  end
  local mode = vehicle:get_mode()
  if mode == MODE_LAND then
    if land_is_climbing(now) then
      if testing then
        report("TEST would STOP MOTORS: LAND is climbing", now)
      else
        arming:disarm()
        report("STOPPED MOTORS: LAND was climbing", now)
      end
    end
    return
  end
  land_low = nil
  local reason = reason_to_land(now, mode)
  if reason ~= nil then
    if testing then
      report("TEST would LAND: " .. reason, now)
    else
      vehicle:set_mode(MODE_LAND)
      report("LAND: " .. reason, now)
    end
  end
end

local function protected()
  local ok, err = pcall(update)
  if not ok then
    gcs:send_text(0, "ALWAYS-LAND script error: " .. tostring(err))
    return protected, 1000
  end
  return protected, 100
end

gcs:send_text(6, "ALWAYS-LAND running")
return protected, 100
