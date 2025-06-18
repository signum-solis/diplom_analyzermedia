package org.example.bear02;

import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.Group;
import javafx.scene.control.*;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;

import java.awt.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.List;

public class HelloController {
    @FXML private ComboBox<String> reportTypeCombo;
    @FXML private TextField keywordField;
    @FXML private ComboBox<String> reportStructureCombo;
    @FXML private DatePicker startDatePicker;
    @FXML private DatePicker endDatePicker;
    @FXML private Button analyzeButton;
    @FXML private ComboBox<String> languageFilterCombo;
    @FXML private CheckBox reliableSourcesCheck;
    @FXML private CheckBox newsAggregatorsCheck;
    @FXML private CheckBox socialNetworksCheck;
    @FXML private CheckBox officialSourcesCheck;
    @FXML private Label totalMentionsLabel;
    @FXML private Label positiveMentionsLabel;
    @FXML private Label negativeMentionsLabel;
    @FXML private Label neutralMentionsLabel;
    @FXML private TextArea reportPreviewArea;
    @FXML private Label statusLabel;

    private static final DateTimeFormatter DATE_FORMATTER = DateTimeFormatter.ofPattern("dd.MM.yyyy");
    private volatile boolean analysisRunning = false;

    @FXML
    private void initialize() {
        // Инициализация значений по умолчанию
        reportStructureCombo.getSelectionModel().selectFirst();
        languageFilterCombo.getSelectionModel().selectFirst();

        new Thread(this::watchMetricsFile).start();



        analyzeButton.setOnAction(event -> {
            if (keywordField.getText().isEmpty()) {
                statusLabel.setText("Введите ключевой запрос!");
                return;
            }

            statusLabel.setText("Анализ запущен...");
            analyzeButton.setDisable(true);
            analysisRunning = true;

            new Thread(() -> {
                runPythonAnalysis();
                Platform.runLater(() -> {
                    analyzeButton.setDisable(false);
                    statusLabel.setText("Анализ завершен");
                });
                analysisRunning = false;
            }).start();
        });



        // Валидация дат
        startDatePicker.valueProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal != null && endDatePicker.getValue() != null && newVal.isAfter(endDatePicker.getValue())) {
                endDatePicker.setValue(newVal);
            }
        });
    }

    private void runPythonAnalysis() {
        try {
            String keyword = keywordField.getText().trim();
            String language = languageFilterCombo.getValue();
            boolean onlyReliable = reliableSourcesCheck.isSelected();
            boolean newsAggregators = newsAggregatorsCheck.isSelected();
            boolean socialNetworks = socialNetworksCheck.isSelected();
            boolean officialSources = officialSourcesCheck.isSelected();
            String dateFrom = startDatePicker.getValue() != null ? startDatePicker.getValue().format(DATE_FORMATTER) : "";
            String dateTo = endDatePicker.getValue() != null ? endDatePicker.getValue().format(DATE_FORMATTER) : "";

            Platform.runLater(reportPreviewArea::clear);

            // Базовые проверки
            if (keyword.isEmpty()) {
                Platform.runLater(() -> statusLabel.setText("Введите ключевой запрос!"));
                return;
            }
            if (dateFrom.isEmpty() || dateTo.isEmpty()) {
                Platform.runLater(() -> statusLabel.setText("Заполните обе даты."));
                return;
            }

            if (!newsAggregatorsCheck.isSelected() && !socialNetworksCheck.isSelected()){
                Platform.runLater(() -> statusLabel.setText("Выберите хотя бы один из источников!"));
                return;
            }

            // Автоматическая блокировка "достоверных" для соцсетей
            if ("Русский".equals(language) || "Английский".equals(language) && socialNetworks) {
                Platform.runLater(() -> {
                    reliableSourcesCheck.setSelected(false);
                    reliableSourcesCheck.setDisable(true);
                });
            } else {
                Platform.runLater(() -> reliableSourcesCheck.setDisable(false));
            }

            // Определяем какие скрипты запускать
            List<Process> processes = new ArrayList<>();

            // Русскоязычные источники
            if ("Русский".equals(language)) {
                // Комбинация 1: Новостные агрегаторы + только достоверные
                if (newsAggregators && onlyReliable) {
                    processes.add(runScript("trueAD/news.py", keyword, dateFrom, dateTo));
                }
                // Комбинация 2: Новостные агрегаторы + официальные источники
                else if (newsAggregators && officialSources) {
                    processes.add(runScript("trueAD/news.py", keyword, dateFrom, dateTo));
                    processes.add(runScript("trueAD/govru.py", keyword, dateFrom, dateTo));
                }
                else if (officialSources) {
                    processes.add(runScript("trueAD/govru.py", keyword, dateFrom, dateTo));

                }
                // Социальные сети
                if (socialNetworks) {
                    processes.add(runScript("source-true/vk.py", keyword, dateFrom, dateTo));

                }
            }
            // Англоязычные источники
            else if ("Английский".equals(language)) {
                if (newsAggregators) {
                    processes.add(runScript("source-true/news.py", keyword, dateFrom, dateTo));
                }
                if (officialSources) {
                    processes.add(runScript("source-true/official.py", keyword, dateFrom, dateTo));
                }
            }


            // Ожидаем завершения всех процессов
            for (Process process : processes) {
                process.waitFor();
            }
            processes.add(runScript("generation/generate.py", reportStructureCombo.getValue(), language));

            // Формируем отчет
            if (!processes.isEmpty()) {
                StringBuilder reportContent = new StringBuilder();
                for (Process process : processes) {
                    try (BufferedReader reader = new BufferedReader(
                            new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
                        String line;
                        while ((line = reader.readLine()) != null) {
                            reportContent.append(line).append("\n");
                        }
                    }
                }

                Platform.runLater(() -> {
                    reportPreviewArea.setText(reportContent.toString());
                    parseResultsFromReport(reportContent.toString());
                    statusLabel.setText("Анализ завершен успешно");
                });
            } else {
                Platform.runLater(() -> statusLabel.setText("Не выбраны источники для анализа"));
            }

        } catch (IOException | InterruptedException e) {
            Platform.runLater(() -> statusLabel.setText("Ошибка запуска анализа: " + e.getMessage()));
        }
    }

    private Process runScript(String scriptName, String... args) throws IOException {
        // Очистка metrics.txt перед анализом
        Files.writeString(Paths.get("metrics.txt"), "", StandardCharsets.UTF_8, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.CREATE);

        String fullPath = "C:\\Users\\danka\\IdeaProjects\\Bear-02\\src\\main\\resources\\org\\example\\bear02\\" + scriptName;
        List<String> command = new ArrayList<>();
        command.add("python");
        command.add(fullPath);
        command.addAll(Arrays.asList(args));

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.redirectErrorStream(true); // объединить stderr в stdout
        Process process = pb.start();

        // Создаем поток для чтения вывода процесса в отдельном потоке, чтобы не блокировать
        new Thread(() -> {
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
                StringBuilder output = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) {
                    final String finalLine = line;
                    // Обновляем UI в потоке JavaFX
                    Platform.runLater(() -> {
                        reportPreviewArea.appendText(finalLine + "\n");
                        // Автоматическая прокрутка к новому содержимому
                        reportPreviewArea.setScrollTop(Double.MAX_VALUE);
                    });
                }
            } catch (IOException e) {
                Platform.runLater(() ->
                        reportPreviewArea.appendText("Ошибка чтения вывода: " + e.getMessage() + "\n"));
            }
        }).start();

        return process;
    }








    private void runGraphScript() {
        try {
            ProcessBuilder pb = new ProcessBuilder("python", "graph.py", keywordField.getText());
            pb.redirectErrorStream(true);
            Process process = pb.start();

            int exitCode = process.waitFor();
            Platform.runLater(() -> {
                if (exitCode == 0) {
                    statusLabel.setText("Граф построен");
                    try {
                        reportPreviewArea.setText(Files.readString(Paths.get("graph_output.txt")));
                    } catch (IOException e) {
                        reportPreviewArea.setText("Граф построен, но не удалось прочитать результат");
                    }
                } else {
                    statusLabel.setText("Ошибка построения графа");
                }
            });

        } catch (IOException | InterruptedException e) {
            Platform.runLater(() -> statusLabel.setText("Ошибка: " + e.getMessage()));
        }
    }

    private void parseResultsFromReport(String reportText) {
        Map<String, Integer> sourceCounts = new HashMap<>();
        int positive = 0, negative = 0, neutral = 0;

        String[] lines = reportText.split("\n");
        for (int i = 0; i < lines.length; i++) {
            String line = lines[i].trim();

            // Ищем количество публикаций по источнику
            if (line.startsWith("- **") && line.contains("публикац")) {
                String[] parts = line.split("\\*\\*");
                if (parts.length >= 3) {
                    String sourceName = parts[1].trim();
                    String countPart = parts[2].replace(":", "").replace("публикаций", "").trim();
                    try {
                        int count = Integer.parseInt(countPart);
                        sourceCounts.put(sourceName, count);
                    } catch (NumberFormatException ignored) {}
                }
            }

            // Обрабатываем строку с тональностью
            if (line.startsWith("-") && line.contains(":") && !line.contains("**")) {
                String[] parts = line.split(":");
                if (parts.length >= 2) {
                    String source = parts[0].substring(2).trim();  // Пропускаем "- "
                    int totalForSource = sourceCounts.getOrDefault(source, 0);

                    // Пример: "Нейтральная: 77%, Негативная: 23%"
                    String sentimentsPart = line.substring(line.indexOf(":") + 1);
                    String[] sentimentEntries = sentimentsPart.split(",");

                    for (String sentimentEntry : sentimentEntries) {
                        sentimentEntry = sentimentEntry.trim();
                        String[] sentSplit = sentimentEntry.split(":");
                        if (sentSplit.length != 2) continue;

                        String sentiment = sentSplit[0].trim();
                        String percentText = sentSplit[1].replace("%", "").trim();

                        try {
                            double percent = Double.parseDouble(percentText);
                            int count = (int) Math.round(totalForSource * (percent / 100.0));

                            switch (sentiment.toLowerCase()) {
                                case "позитивная", "положительная", "positive" -> positive += count;
                                case "нейтральная", "neutral" -> neutral += count;
                                case "негативная", "negative" -> negative += count;
                            }
                        } catch (NumberFormatException ignored) {}
                    }
                }
            }
        }

        int total = positive + negative + neutral;
        updateStatistics(total, positive, negative, neutral);
    }


    private int extractPercentage(String line, String sentiment) {
        try {
            int idx = line.indexOf(sentiment);
            if (idx == -1) return 0;
            int percentIdx = line.indexOf('%', idx);
            if (percentIdx == -1) return 0;
            String numberPart = line.substring(idx + sentiment.length() + 1, percentIdx).trim();
            return (int) Math.round(Double.parseDouble(numberPart)); // Округляем процент
        } catch (Exception e) {
            return 0;
        }
    }

    private void watchMetricsFile() {
        Path metricsPath = Paths.get("C:\\Users\\danka\\IdeaProjects\\Bear-02\\metrics.txt");
        try {
            WatchService watchService = FileSystems.getDefault().newWatchService();
            metricsPath.getParent().register(watchService, StandardWatchEventKinds.ENTRY_MODIFY);

            while (true) {
                WatchKey key = watchService.take();
                for (WatchEvent<?> event : key.pollEvents()) {
                    Path changed = (Path) event.context();
                    if (changed.endsWith("metrics.txt")) {
                        Thread.sleep(1000); // Даем файлу немного времени на запись
                        Platform.runLater(this::updateMetricsFromFile);
                    }
                }
                key.reset();
            }
        } catch (IOException | InterruptedException e) {
            Platform.runLater(() -> statusLabel.setText("Ошибка слежения за metrics.txt: " + e.getMessage()));
        }
    }

    private void updateMetricsFromFile() {
        try {
            List<String> lines = Files.readAllLines(Paths.get("metrics.txt"), StandardCharsets.UTF_8);

            int total = 0, positive = 0, negative = 0, neutral = 0;

            for (String line : lines) {
                if (line.startsWith("total_mentions:")) {
                    total = Integer.parseInt(line.split(":")[1].trim());
                } else if (line.startsWith("sentiment_distribution:")) {
                    String data = line.substring(line.indexOf("{") + 1, line.lastIndexOf("}"));
                    String[] entries = data.split(",");

                    for (String entry : entries) {
                        String[] parts = entry.trim().replace("\"", "").split(":");
                        if (parts.length == 2) {
                            String sentiment = parts[0].trim().toLowerCase();
                            int count = Integer.parseInt(parts[1].trim());

                            switch (sentiment) {
                                case "позитивная", "positive" -> positive = count;
                                case "негативная", "negative" -> negative = count;
                                case "нейтральная", "neutral" -> neutral = count;
                            }
                        }
                    }
                }
            }

            updateStatistics(total, positive, negative, neutral);
        } catch (IOException | NumberFormatException e) {
            Platform.runLater(() -> statusLabel.setText("Ошибка чтения metrics.txt: " + e.getMessage()));
        }
    }




    private void updateStatistics(int total, int positive, int negative, int neutral) {
        totalMentionsLabel.setText(String.valueOf(total));
        positiveMentionsLabel.setText(String.valueOf(positive));
        negativeMentionsLabel.setText(String.valueOf(negative));
        neutralMentionsLabel.setText(String.valueOf(neutral));
    }

    @FXML
    private void generatePdfReport() {
        String keyword = keywordField.getText().trim();
        String language = languageFilterCombo.getValue();
        String template = reportStructureCombo.getValue();

        if (keyword.isEmpty()) {
            statusLabel.setText("Введите ключевое слово!");
            return;
        }

        statusLabel.setText("Генерация PDF отчета...");

        new Thread(() -> {
            try {
                ProcessBuilder pb = new ProcessBuilder(
                        "python",
                        "C:\\Users\\danka\\IdeaProjects\\Bear-02\\src\\main\\resources\\org\\example\\bear02\\generate_pdf.py",
                        template,
                        keyword,
                        language
                );

                pb.redirectErrorStream(true);
                Process process = pb.start();

                // Для сбора вывода
                StringBuilder output = new StringBuilder();
                String savedPath = null;

                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        output.append(line).append("\n");

                        // Ищем строку с путем сохранения
                        if (line.startsWith("SAVED_TO:")) {
                            savedPath = line.substring("SAVED_TO:".length()).trim();
                        }
                    }
                }

                int exitCode = process.waitFor();
                final String finalSavedPath = savedPath; // Фиксируем значение
                Platform.runLater(() -> {
                    if (exitCode == 0) {
                        if (finalSavedPath != null) {
                            statusLabel.setText("PDF сохранен: " + finalSavedPath);
                            reportPreviewArea.appendText("\nPDF отчет сохранен: " + finalSavedPath + "\n");
                        } else {
                            statusLabel.setText("PDF создан, но путь не определен");
                        }
                    } else {
                        statusLabel.setText("Ошибка при создании PDF. Код: " + exitCode);
                        reportPreviewArea.appendText("\nОшибка генерации PDF:\n" + output.toString() + "\n");
                    }
                });
            } catch (IOException | InterruptedException e) {
                Platform.runLater(() -> {
                    statusLabel.setText("Ошибка: " + e.getMessage());
                    reportPreviewArea.appendText("\nОшибка выполнения: " + e.getMessage() + "\n");
                });
            }
        }).start();
    }

    @FXML
    private void generateMarkdownReport() {
        String keyword = keywordField.getText().trim();
        String language = languageFilterCombo.getValue();
        String template = reportStructureCombo.getValue();

        if (keyword.isEmpty()) {
            statusLabel.setText("Введите ключевое слово!");
            return;
        }

        statusLabel.setText("Генерация Markdown отчета...");

        new Thread(() -> {
            try {
                // Запускаем Python скрипт для генерации Markdown
                ProcessBuilder pb = new ProcessBuilder(
                        "python",
                        "C:\\Users\\danka\\IdeaProjects\\Bear-02\\src\\main\\resources\\org\\example\\bear02\\generate_mardown.py",
                        template,
                        keyword,
                        language
                );

                pb.redirectErrorStream(true);
                Process process = pb.start();

                // Чтение вывода скрипта (для отладки)
                BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                String line;
                while ((line = reader.readLine()) != null) {
                    System.out.println(line);
                }

                int exitCode = process.waitFor();
                Platform.runLater(() -> {
                    if (exitCode == 0) {
                        statusLabel.setText("Markdown отчет успешно создан!");
                        // Можно добавить открытие файла в редакторе по умолчанию
                        // Desktop.getDesktop().open(new File("report.md"));
                    } else {
                        statusLabel.setText("Ошибка при создании Markdown отчета");
                    }
                });
            } catch (IOException | InterruptedException e) {
                Platform.runLater(() ->
                        statusLabel.setText("Ошибка: " + e.getMessage()));
                e.printStackTrace();
            }
        }).start();
    }

    @FXML
    private void generateHtmlReport() {
        String keyword = keywordField.getText().trim();
        String language = languageFilterCombo.getValue();

        if (keyword.isEmpty()) {
            statusLabel.setText("Введите ключевое слово!");
            return;
        }

        statusLabel.setText("Генерация HTML отчета...");
        reportPreviewArea.appendText("\nНачало генерации HTML отчета...\n");

        new Thread(() -> {
            try {
                // Запускаем Python скрипт для генерации HTML
                ProcessBuilder pb = new ProcessBuilder(
                        "python",
                        "C:\\Users\\danka\\IdeaProjects\\Bear-02\\src\\main\\resources\\org\\example\\bear02\\generat_html.py",
                        keyword,
                        language
                );

                pb.redirectErrorStream(true);
                Process process = pb.start();

                // Чтение вывода скрипта в реальном времени
                BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                StringBuilder output = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line).append("\n");
                    String currentLine = line;
                    Platform.runLater(() -> reportPreviewArea.appendText(currentLine + "\n"));
                }

                int exitCode = process.waitFor();

                Platform.runLater(() -> {
                    if (exitCode == 0) {
                        statusLabel.setText("HTML отчет успешно создан");
                        reportPreviewArea.appendText("\nГенерация HTML отчета завершена успешно!\n");
                    } else {
                        statusLabel.setText("Ошибка при создании HTML. Код: " + exitCode);
                        reportPreviewArea.appendText("\nОшибка генерации HTML:\n" + output.toString() + "\n");
                    }
                });
            } catch (IOException | InterruptedException e) {
                Platform.runLater(() -> {
                    statusLabel.setText("Ошибка: " + e.getMessage());
                    reportPreviewArea.appendText("\nОшибка выполнения: " + e.getMessage() + "\n");
                });
            }
        }).start();
    }

    private String extractSavedPath(String output) {
        // Ищем в выводе строку с путем к файлу
        String marker = "HTML отчет сохранен как ";
        int start = output.indexOf(marker);
        if (start >= 0) {
            start += marker.length();
            int end = output.indexOf("\n", start);
            return end >= 0 ? output.substring(start, end) : output.substring(start);
        }
        return null;
    }

    private void openHtmlReport(String path) {
        try {
            File htmlFile = new File(path);
            if (htmlFile.exists()) {
                Desktop.getDesktop().browse(htmlFile.toURI());
            } else {
                statusLabel.setText("Файл не найден: " + path);
            }
        } catch (IOException e) {
            Platform.runLater(() -> {
                statusLabel.setText("Ошибка открытия: " + e.getMessage());
                reportPreviewArea.appendText("\nОшибка при открытии отчета: " + e.getMessage() + "\n");
            });
        }
    }
}