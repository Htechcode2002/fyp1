-- IA-Vision System - Database Export Template
-- Use this file to manually create the database structure if needed.

CREATE DATABASE IF NOT EXISTS `test`;
USE `test`;

-- Table structure for table `crossing_events`
CREATE TABLE IF NOT EXISTS `crossing_events` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `video_id` VARCHAR(255),
    `location` VARCHAR(255),
    `line_name` VARCHAR(255),
    `count_left` INT DEFAULT 0,
    `count_right` INT DEFAULT 0,
    `clothing_color` VARCHAR(255),
    `gender` VARCHAR(50),
    `age` VARCHAR(50),
    `mask_status` VARCHAR(50),
    `handbag` TINYINT DEFAULT 0,
    `backpack` TINYINT DEFAULT 0,
    `timestamp` DATETIME
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Note: The system will automatically add columns if they are missing at runtime.
