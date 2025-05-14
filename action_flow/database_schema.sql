-- MySQL dump 10.13  Distrib 8.0.42, for Win64 (x86_64)
--
-- Host: localhost    Database: my_database
-- ------------------------------------------------------
-- Server version	8.0.42

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `flight_booking_data`
--

DROP TABLE IF EXISTS `flight_booking_data`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `flight_booking_data` (
  `User_ID` int DEFAULT NULL,
  `Flight_ID` varchar(10) DEFAULT NULL,
  `Booking_Date` date DEFAULT NULL,
  `Travel_Date` date DEFAULT NULL,
  `PNR` varchar(10) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `flight_booking_data`
--

LOCK TABLES `flight_booking_data` WRITE;
/*!40000 ALTER TABLE `flight_booking_data` DISABLE KEYS */;
INSERT INTO `flight_booking_data` VALUES (101,'6E2013','2025-04-11','2025-05-09','BIN1001'),(105,'AI803','2025-04-13','2025-04-25','BIN1003'),(101,'SG102','2025-04-15','2025-04-27','BIN1005'),(109,'UK811','2025-04-16','2025-04-28','BIN1006'),(107,'AI675','2025-04-17','2025-04-29','BIN1007');
/*!40000 ALTER TABLE `flight_booking_data` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `flight_data`
--

DROP TABLE IF EXISTS `flight_data`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `flight_data` (
  `Flight_Number` varchar(10) NOT NULL,
  `Airline` varchar(50) DEFAULT NULL,
  `Departure_City` varchar(50) DEFAULT NULL,
  `Arrival_City` varchar(50) DEFAULT NULL,
  `Departure_Time` time DEFAULT NULL,
  `Arrival_Time` time DEFAULT NULL,
  `Duration` varchar(10) DEFAULT NULL,
  `Base_Fair` int DEFAULT NULL,
  PRIMARY KEY (`Flight_Number`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `flight_data`
--

LOCK TABLES `flight_data` WRITE;
/*!40000 ALTER TABLE `flight_data` DISABLE KEYS */;
INSERT INTO `flight_data` VALUES ('6E2003','SkyWings','Ahmedabad','Delhi','05:45:00','07:30:00','1h 45m',9500),('6E2013','SkyWings','Pune','Bangalore','07:25:00','09:00:00','1h 35m',5000),('6E2023','SkyWings','Bangalore','Kolkata','07:15:00','09:55:00','2h 40m',6500),('6E2110','SkyWings','Hyderabad','Bangalore','12:30:00','13:40:00','1h 10m',6000),('6E330','SkyWings','Kochi','Bangalore','06:20:00','07:30:00','1h 10m',6100),('6E750','SkyWings','Kolkata','Guwahati','13:30:00','14:45:00','1h 15m',7800),('6E775','SkyWings','Pune','Hyderabad','18:00:00','19:20:00','1h 20m',5800),('AI101','SkyWings','Delhi','Mumbai','06:00:00','08:05:00','2h 5m',7000),('AI441','SkyWings','Bhopal','Delhi','09:45:00','11:15:00','1h 30m',5800),('AI502','SkyWings','Mumbai','Nagpur','07:10:00','08:40:00','1h 30m',4500),('AI675','SkyWings','Delhi','Pune','18:00:00','20:00:00','2h 0m',8800),('AI803','SkyWings','Delhi','Hyderabad','10:00:00','12:10:00','2h 10m',7000),('SG102','SkyWings','Kolkata','Delhi','14:00:00','16:30:00','2h 30m',5500),('SG247','SkyWings','Surat','Delhi','14:15:00','16:05:00','1h 50m',6500),('SG301','SkyWings','Mumbai','Delhi','09:00:00','11:10:00','2h 10m',8000),('SG422','SkyWings','Patna','Kolkata','13:00:00','14:10:00','1h 10m',3500),('SG509','SkyWings','Jaipur','Mumbai','11:30:00','13:20:00','1h 50m',4500),('UK120','SkyWings','Chennai','Mumbai','08:20:00','10:45:00','2h 25m',5000),('UK811','SkyWings','Mumbai','Goa','16:10:00','17:25:00','1h 15m',5400),('UK831','SkyWings','Lucknow','Delhi','07:00:00','08:10:00','1h 10m',6500),('UK946','SkyWings','Delhi','Chandigarh','15:45:00','16:45:00','1h 0m',4500);
/*!40000 ALTER TABLE `flight_data` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `user_data`
--

DROP TABLE IF EXISTS `user_data`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `user_data` (
  `ID` int NOT NULL,
  `Name` varchar(100) DEFAULT NULL,
  `Mobile_Number` varchar(15) DEFAULT NULL,
  `Email` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`ID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `user_data`
--

LOCK TABLES `user_data` WRITE;
/*!40000 ALTER TABLE `user_data` DISABLE KEYS */;
INSERT INTO `user_data` VALUES (101,'Amit Datta','9960907555','AD1989@gmail.com'),(102,'Satgur','9960907556','ST1989@gmail.com'),(103,'Pankaj Goyal','9960907557','PG1989@gmail.com'),(104,'Sree','9960907558','Sree1989@gmail.com'),(105,'Aditi','9960907559','Aditi1989@gmail.com'),(106,'Harshal','9960907560','Harshal1989@gmail.com'),(107,'Sajeesh','9960907561','Saj1989@gmail.com'),(108,'Raghu','9960907562','Raghu1989@gmail.com'),(109,'Vikas','9960907563','Vikas1989@gmail.com'),(110,'Anusha','9960907564','Anusha1989@gmail.com');
/*!40000 ALTER TABLE `user_data` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-04-30 15:13:58
