CREATE TABLE `norovirus_nowcasts`(
  `id`            INT(11)      NOT NULL PRIMARY KEY AUTO_INCREMENT,
  `target`        VARCHAR(32)  NOT NULL                           ,
  `epiweek`       INT(11)      NOT NULL                           ,
  `location`      VARCHAR(12)  NULL                               ,
  `value`         FLOAT        NOT NULL                           ,
  `std`         FLOAT        NOT NULL                           ,
  UNIQUE KEY `entry` (`target`, `epiweek`, `location`),
  KEY `target` (`target`),
  KEY `epiweek` (`epiweek`),
  KEY `location` (`location`)
);
