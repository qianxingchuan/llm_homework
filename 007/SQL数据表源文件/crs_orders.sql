/*
 Navicat MySQL Data Transfer

 Source Server         : 阿里云
 Source Server Type    : MySQL
 Source Server Version : 80025 (8.0.25)
 Source Host           : rm-uf6z891lon6dxuqblqo.mysql.rds.aliyuncs.com:3306
 Source Schema         : life_insurance

 Target Server Type    : MySQL
 Target Server Version : 80025 (8.0.25)
 File Encoding         : 65001

 Date: 24/04/2025 22:44:33
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for crs_orders
-- ----------------------------
DROP TABLE IF EXISTS `crs_orders`;
CREATE TABLE `crs_orders`  (
  `order_time` datetime NULL DEFAULT NULL,
  `crs_user_id` bigint NULL DEFAULT NULL,
  `eco_main_order_id` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL,
  `channel` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL,
  `status_code` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL,
  `hotel_code` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL,
  `reserved_roomtype_code` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL,
  `actual_roomtype_code` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL,
  `rate_code` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL,
  `rooms` bigint NULL DEFAULT NULL,
  `RNs` bigint NULL DEFAULT NULL,
  `adults` bigint NULL DEFAULT NULL,
  `children` bigint NULL DEFAULT NULL,
  `no_guests` bigint NULL DEFAULT NULL,
  `total_revenue` double NULL DEFAULT NULL,
  `city` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL,
  `province` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL,
  `age` bigint NULL DEFAULT NULL,
  `gender` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL,
  `arrival` datetime NULL DEFAULT NULL,
  `departure` datetime NULL DEFAULT NULL,
  `event_timestamp` bigint NULL DEFAULT NULL,
  `eventid` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL,
  `offset` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL,
  `productid` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL
) ENGINE = InnoDB AUTO_INCREMENT = 11 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of crs_orders
-- ----------------------------
INSERT INTO `crs_orders` VALUES ('2023-02-21 01:44:00', 16258, 'ORD402692', 'TAS', 'checkout', 'NUPEK11', 'D2Q', 'D2Q', 'TABAR', 2, 2, 3, 3, 6, 478.6, '北京市', '北京', 18, 'F', '2023-02-21 01:44:00', '2023-02-23 01:44:00', 1676915040, 'EVT869962', 'OFF22635', 'App');
INSERT INTO `crs_orders` VALUES ('2023-03-11 04:38:22', 73100, 'ORD203772', 'TAS', 'checkout', 'ULPEK10', 'D2Q', 'D2Q', 'FLI618G', 2, 4, 4, 2, 6, 838.36, '上海市', '上海', 32, 'M', '2023-03-11 04:38:22', '2023-03-12 04:38:22', 1678480702, 'EVT508794', 'OFF86528', 'App');
INSERT INTO `crs_orders` VALUES ('2023-01-08 09:03:54', 66608, 'ORD517695', 'FLI', 'checkout', 'ULPEK10', 'D2Q', 'D2Q', 'FLI618G', 4, 4, 1, 0, 1, 958.13, '北京市', '北京', 49, 'M', '2023-01-08 09:03:54', '2023-01-09 09:03:54', 1673139834, 'EVT675098', 'OFF36718', 'App');
INSERT INTO `crs_orders` VALUES ('2023-12-03 02:32:55', 52955, 'ORD612940', 'TAS', 'checkout', 'ULPEK10', 'D2Q', 'D2Q', 'TABAR', 1, 1, 4, 0, 4, 256.91, '厦门市', '福建省', 55, 'M', '2023-12-03 02:32:55', '2023-12-05 02:32:55', 1701541975, 'EVT764058', 'OFF85663', 'WeChat Mini-program');
INSERT INTO `crs_orders` VALUES ('2023-09-27 14:14:54', 69732, 'ORD360068', 'TAS', 'checkout', 'NUPEK11', 'D2Q', 'D2Q', 'TABAR', 3, 3, 3, 2, 5, 462.3, '北京市', '北京', 26, 'F', '2023-09-27 14:14:54', '2023-09-29 14:14:54', 1695795294, 'EVT531411', 'OFF72880', 'WeChat Mini-program');
INSERT INTO `crs_orders` VALUES ('2023-05-25 22:05:02', 89525, 'ORD496395', 'TAS', 'checkout', 'NUPEK11', 'B2Q', 'B2Q', 'FLI618G', 5, 10, 2, 1, 3, 106.59, '重庆市', '重庆', 42, 'F', '2023-05-25 22:05:02', '2023-05-27 22:05:02', 1685023502, 'EVT596135', 'OFF99330', 'App');
INSERT INTO `crs_orders` VALUES ('2023-10-18 12:12:55', 44275, 'ORD891640', 'FLI', 'checkout', 'ULPEK10', 'B2Q', 'B2Q', 'TABAR', 5, 5, 3, 0, 3, 836.92, '北京市', '北京', 19, 'M', '2023-10-18 12:12:55', '2023-10-20 12:12:55', 1697602375, 'EVT277282', 'OFF22047', 'App');
INSERT INTO `crs_orders` VALUES ('2023-08-03 11:09:58', 96029, 'ORD166072', 'TAS', 'checkout', 'NUPEK11', 'B2Q', 'B2Q', 'FLI618G', 1, 1, 1, 3, 4, 761.61, '北京市', '北京', 50, 'F', '2023-08-03 11:09:58', '2023-08-04 11:09:58', 1691032198, 'EVT118054', 'OFF56708', 'App');
INSERT INTO `crs_orders` VALUES ('2023-05-22 05:33:59', 87818, 'ORD268727', 'TAS', 'checkout', 'NUPEK11', 'B2Q', 'B2Q', 'FLI618G', 3, 6, 4, 1, 5, 607.45, '北京市', '北京', 55, 'M', '2023-05-22 05:33:59', '2023-05-23 05:33:59', 1684704839, 'EVT264762', 'OFF72199', 'WeChat Mini-program');
INSERT INTO `crs_orders` VALUES ('2023-02-17 04:07:52', 43495, 'ORD545010', 'FLI', 'checkout', 'NUPEK11', 'D2Q', 'D2Q', 'TABAR', 2, 4, 1, 2, 3, 309.77, '青岛市', '山东省', 18, 'M', '2023-02-17 04:07:52', '2023-02-18 04:07:52', 1676578072, 'EVT952941', 'OFF70458', 'WeChat Mini-program');

SET FOREIGN_KEY_CHECKS = 1;
