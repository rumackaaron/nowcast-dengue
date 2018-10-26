CREATE TABLE `norovirus_sensors`(
  `id`            INT(11)     NOT NULL PRIMARY KEY AUTO_INCREMENT,
  -- `target_source` VARCHAR(8)  NOT NULL                           ,
  -- `target_column` VARCHAR(8)  NOT NULL                           ,
  `name`          VARCHAR(8)  NOT NULL                           ,
  `epiweek`       INT(11)     NOT NULL                           ,
  `location`      VARCHAR(12) NULL                               ,
  `value`         FLOAT       NOT NULL                           ,
  -- UNIQUE KEY `entry` (`target_source`, `target_column`, `name`, `epiweek`, `location`),
  UNIQUE KEY `entry` (`name`, `epiweek`, `location`),
  -- KEY `sensor` (`target_source`, `target_column`, `name`),
  KEY `name` (`name`),
  KEY `epiweek` (`epiweek`),
  KEY `location` (`location`)
);
