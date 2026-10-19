import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// A persisted app-language preference.  Screens can read [locale] for text
/// direction and API clients pass [chatLanguageCode] to the AI assistant.
class AppLanguage {
  static const _key = 'staybuddy_language';
  static final ValueNotifier<Locale> locale = ValueNotifier(const Locale('en'));

  static Future<void> load() async {
    final preferences = await SharedPreferences.getInstance();
    locale.value = Locale(preferences.getString(_key) == 'ur' ? 'ur' : 'en');
  }

  static Future<void> set(String code) async {
    final normalized = code == 'ur' ? 'ur' : 'en';
    locale.value = Locale(normalized);
    final preferences = await SharedPreferences.getInstance();
    await preferences.setString(_key, normalized);
  }

  static String get chatLanguageCode => locale.value.languageCode;
  static bool get isUrdu => locale.value.languageCode == 'ur';
}
