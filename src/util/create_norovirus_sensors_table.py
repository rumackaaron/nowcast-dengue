CREATE TABLE `norovirus_sensors`(
  `id`            INT(11)      NOT NULL PRIMARY KEY AUTO_INCREMENT,
  `target`        VARCHAR(32)  NOT NULL                           ,
  `name`          VARCHAR(8)   NOT NULL                           ,
  `epiweek`       INT(11)      NOT NULL                           ,
  `location`      VARCHAR(12)  NULL                               ,
  `value`         FLOAT        NOT NULL                           ,
  UNIQUE KEY `entry` (`target`, `name`, `epiweek`, `location`),
  KEY `sensor` (`target`, `name`),
  KEY `epiweek` (`epiweek`),
  KEY `location` (`location`)
);
