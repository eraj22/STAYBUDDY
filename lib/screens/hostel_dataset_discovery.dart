import 'dart:async';
import 'dart:convert';
import 'dart:math';

import 'package:csv/csv.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_map/flutter_map.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;
import 'package:latlong2/latlong.dart';
import '../routes.dart';
import '../services/location_service.dart';
import '../widgets/responsive_shell.dart';
import 'package:url_launcher/url_launcher.dart';
import 'student_booking_info_screen.dart';
import 'ai_recommendation_screen.dart';
import 'chatbot_screen.dart';
import 'hostel_detail_screen.dart';

// ─── Helpers ────────────────────────────────────────────────────────────────

String normalizeAmenity(String value) {
  final key = value
      .toLowerCase()
      .trim()
      .replaceAll('&', 'and')
      .replaceAll(RegExp(r'\s+'), ' ')
      .replaceAll('-', ' ');

  const aliases = {
    'wifi': 'wifi', 'wi fi': 'wifi', 'wi-fi': 'wifi', 'internet': 'wifi',
    'ac': 'ac', 'air conditioning': 'ac',
    'cctv': 'cctv',
    'security': 'security guard', 'security guard': 'security guard',
    'study': 'study room', 'study room': 'study room',
    'hot water': 'hot water',
    'common room': 'common room',
    'gym': 'gym', 'parking': 'parking', 'iron': 'iron',
    'generator': 'generator', 'library': 'library', 'laundry': 'laundry',
    'cafeteria': 'cafeteria', 'prayer room': 'prayer room',
    'rooftop access': 'rooftop access', 'microwave': 'microwave',
    'water cooler': 'water cooler', 'refrigerator': 'refrigerator',
    'lounge': 'lounge',
  };
  return aliases[key] ?? key;
}

String prettyAmenity(String value) {
  switch (normalizeAmenity(value)) {
    case 'wifi': return 'WiFi';
    case 'ac': return 'AC';
    case 'cctv': return 'CCTV';
    case 'security guard': return 'Security Guard';
    case 'study room': return 'Study Room';
    case 'hot water': return 'Hot Water';
    case 'common room': return 'Common Room';
    case 'prayer room': return 'Prayer Room';
    case 'rooftop access': return 'Rooftop Access';
    case 'water cooler': return 'Water Cooler';
    default:
      return normalizeAmenity(value).split(' ')
          .map((w) => w.isEmpty ? w : '${w[0].toUpperCase()}${w.substring(1)}')
          .join(' ');
  }
}

String normalizeRoomType(String value) {
  final key = value.toLowerCase().trim().replaceAll(RegExp(r'\s+'), ' ');
  const aliases = {
    'single': 'single', 'double': 'double',
    'triple': 'triple', 'dormitory': 'dormitory', 'dorm': 'dormitory',
  };
  return aliases[key] ?? key;
}

String prettyRoomType(String value) {
  switch (normalizeRoomType(value)) {
    case 'single': return 'Single';
    case 'double': return 'Double';
    case 'triple': return 'Triple';
    case 'dormitory': return 'Dormitory';
    default:
      return normalizeRoomType(value).split(' ')
          .map((w) => w.isEmpty ? w : '${w[0].toUpperCase()}${w.substring(1)}')
          .join(' ');
  }
}

enum GenderFilter { all, girls, boys }
enum FoodTypeFilter { all, veg, nonVeg, both }

// ─── Theme ──────────────────────────────────────────────────────────────────

const _teal        = Color(0xFF0A6B6E);
const _tealDark    = Color(0xFF074F51);
const _tealLight   = Color(0xFF1D9E75);
const _tealMint    = Color(0xFF5DCAA5);
const _bg          = Color(0xFFF5F3EE);
const _card        = Color(0xFFFFFFFF);
const _border      = Color(0xFFE0DDD6);
const _textPrimary = Color(0xFF0A3D3F);
const _textMuted   = Color(0xFF888780);

// ─── Models ──────────────────────────────────────────────────────────────────

class PlaceSuggestion {
  final String title, subtitle;
  final LatLng point;
  PlaceSuggestion({required this.title, required this.subtitle, required this.point});
}

class Hostel {
  final String id, name, type, area, city, address;
  final double lat, lng;
  final int? singlePrice, doublePrice, dormPrice;
  final List<String> amenities, roomTypesAvailable;
  final bool electricityIncluded, mealIncluded;
  final String? foodType;
  final double? foodRating, overallRating;
  final int? totalReviews;

  Hostel({
    required this.id, required this.name, required this.type,
    required this.area, required this.city, required this.address,
    required this.lat, required this.lng, required this.amenities,
    required this.electricityIncluded, required this.mealIncluded,
    required this.foodType, required this.foodRating,
    required this.roomTypesAvailable, required this.overallRating,
    required this.totalReviews,
    this.singlePrice, this.doublePrice, this.dormPrice,
  });

  LatLng get point => LatLng(lat, lng);

  int? get minPrice {
    final prices = <int>[
      if (singlePrice != null && singlePrice! > 0) singlePrice!,
      if (doublePrice != null && doublePrice! > 0) doublePrice!,
      if (dormPrice != null && dormPrice! > 0) dormPrice!,
    ];
    if (prices.isEmpty) return null;
    prices.sort();
    return prices.first;
  }

  bool hasAmenity(String a) =>
      amenities.any((x) => normalizeAmenity(x) == normalizeAmenity(a));

  List<String> matchedAmenities(Set<String> sel) =>
      amenities.where((a) => sel.any((s) => normalizeAmenity(s) == normalizeAmenity(a))).toList();

  List<String> otherAmenities(Set<String> sel) =>
      amenities.where((a) => !sel.any((s) => normalizeAmenity(s) == normalizeAmenity(a))).toList();

  bool hasRoomType(String rt) =>
      roomTypesAvailable.any((r) => normalizeRoomType(r) == normalizeRoomType(rt));
}

// ─── Main Widget ─────────────────────────────────────────────────────────────

class HostelDatasetDiscovery extends StatefulWidget {
  final double initialLat, initialLng;
  final bool locationEnabled;

  const HostelDatasetDiscovery({
    super.key,
    required this.initialLat,
    required this.initialLng,
    required this.locationEnabled,
  });

  @override
  State<HostelDatasetDiscovery> createState() => _HostelDatasetDiscoveryState();
}

class _HostelDatasetDiscoveryState extends State<HostelDatasetDiscovery>
    with TickerProviderStateMixin {

  final MapController _mapController = MapController();
  final TextEditingController _searchController = TextEditingController();
  final FocusNode _searchFocus = FocusNode();

  late bool _locationEnabled;
  late LatLng _currentCenter;
  double _zoom = 14;

  // Filters
  double _radiusKm = 3;
  GenderFilter _genderFilter = GenderFilter.all;
  bool _budgetFilterOn = false;
  int _budgetMin = 15000, _budgetMax = 35000;
  bool _includeUnknownPrice = true;
  final Set<String> _selectedAmenities = {};
  List<String> _amenityOptions = [];
  bool _requireElectricityIncluded = false, _requireMealsIncluded = false;
  FoodTypeFilter _foodTypeFilter = FoodTypeFilter.all;
  bool _foodRatingFilterOn = false;
  double _minFoodRating = 3.0;
  final Set<String> _selectedRoomTypes = {};
  List<String> _roomTypeOptions = [];
  bool _hostelRatingFilterOn = false;
  double _minHostelRating = 3.5;
  bool _reviewsFilterOn = false;
  int _minReviews = 20;

  // Data
  bool _loadingDataset = true;
  List<Hostel> _allHostels = [];
  List<Hostel> _filteredHostels = [];
  bool _loading = false;
  String? _statusText;

  // Search
  Timer? _suggestDebounce;
  bool _showSuggestions = false, _loadingSuggestions = false;
  List<PlaceSuggestion> _suggestions = [];
  bool _ignoreNextTextChange = false;
  Marker? _searchMarker;

  // Animation controllers
  late final AnimationController _bottomSheetAnim;
  late final AnimationController _fabAnim;
  late final Animation<double> _bottomSheetSlide;
  late final Animation<double> _fabScale;

  // Selected hostel for quick preview
  Hostel? _selectedHostel;
  late final AnimationController _previewAnim;
  late final Animation<double> _previewSlide;

  @override
  void initState() {
    super.initState();
    _locationEnabled = widget.locationEnabled;
    _currentCenter = LatLng(widget.initialLat, widget.initialLng);
    _searchController.addListener(_onSearchTextChanged);

    _bottomSheetAnim = AnimationController(
      vsync: this, duration: const Duration(milliseconds: 600),
    );
    _bottomSheetSlide = CurvedAnimation(
      parent: _bottomSheetAnim, curve: Curves.easeOutCubic,
    );

    _fabAnim = AnimationController(
      vsync: this, duration: const Duration(milliseconds: 500),
    );
    _fabScale = CurvedAnimation(parent: _fabAnim, curve: Curves.elasticOut);

    _previewAnim = AnimationController(
      vsync: this, duration: const Duration(milliseconds: 350),
    );
    _previewSlide = CurvedAnimation(parent: _previewAnim, curve: Curves.easeOutCubic);

    _initDataset();
    if (_locationEnabled) _goToMyLocation();

    Future.delayed(const Duration(milliseconds: 300), () {
      _bottomSheetAnim.forward();
      _fabAnim.forward();
    });
  }

  @override
  void dispose() {
    _suggestDebounce?.cancel();
    _searchController.removeListener(_onSearchTextChanged);
    _searchController.dispose();
    _searchFocus.dispose();
    _bottomSheetAnim.dispose();
    _fabAnim.dispose();
    _previewAnim.dispose();
    super.dispose();
  }

  // ─── Data ─────────────────────────────────────────────────────────────────

  Future<void> _initDataset() async {
    setState(() => _loadingDataset = true);
    try {
      final csvString = await rootBundle.loadString('assets/data/hostels_featured.csv');
      final rows = const CsvToListConverter(eol: '\n', shouldParseNumbers: false)
          .convert(csvString);

      if (rows.isEmpty) {
        setState(() { _allHostels = []; _filteredHostels = []; _loadingDataset = false; });
        return;
      }

      final header = rows.first.map((e) => e.toString()).toList();
      int idx(String col) => header.indexOf(col);

      final iId = idx('hostel_id'), iName = idx('hostel_name'),
            iType = idx('hostel_type'), iArea = idx('area'),
            iCity = idx('city'), iAddr = idx('hostel_address'),
            iLat = idx('latitude'), iLng = idx('longitude'),
            iSingle = idx('single_room_price'), iDouble = idx('double_room_price'),
            iDorm = idx('dorm_room_price'), iAmenities = idx('amenities'),
            iWifi = idx('has_wifi'), iGym = idx('has_gym'),
            iStudyRoom = idx('has_study_room'), iCafeteria = idx('has_cafeteria'),
            iLaundry = idx('has_laundry'), iAc = idx('has_ac'),
            iGenerator = idx('has_generator'), iSecurityGuard = idx('has_security_guard'),
            iCctv = idx('has_cctv'), iHotWater = idx('has_hot_water'),
            iLibrary = idx('has_library'), iParking = idx('has_parking'),
            iPrayerRoom = idx('has_prayer_room'), iCommonRoom = idx('has_common_room'),
            iElectricity = idx('electricity_included'), iMeal = idx('meal_included'),
            iFoodType = idx('food_type'), iFoodRating = idx('food_rating'),
            iRoomTypes = idx('room_types_available'), iRating = idx('overall_rating'),
            iReviews = idx('total_reviews');

      int? parseInt(dynamic v) {
        final s = v?.toString().trim() ?? '';
        return s.isEmpty ? null : int.tryParse(s);
      }
      double? parseDouble(dynamic v) {
        final s = v?.toString().trim() ?? '';
        return s.isEmpty ? null : double.tryParse(s);
      }
      List<String> parseList(dynamic v) {
        if (v == null) return [];
        final raw = v.toString().trim();
        if (raw.isEmpty) return [];
        try {
          final decoded = jsonDecode(raw);
          if (decoded is List) return decoded.map((e) => e.toString().trim()).where((e) => e.isNotEmpty).toList();
        } catch (_) {}
        return raw.replaceAll('[', '').replaceAll(']', '').replaceAll('"', '').split(',')
            .map((e) => e.trim()).where((e) => e.isNotEmpty).toList();
      }
      bool isYes(dynamic v) {
        final s = v?.toString().trim().toLowerCase() ?? '';
        return s == '1' || s == 'true' || s == 'yes';
      }

      final hostels = <Hostel>[];
      final amenitySet = <String>{}, roomTypeSet = <String>{};

      for (var r = 1; r < rows.length; r++) {
        final row = rows[r];
        if (iLat < 0 || iLng < 0 || iName < 0 || iId < 0) continue;
        final lat = parseDouble(row[iLat]), lng = parseDouble(row[iLng]);
        if (lat == null || lng == null) continue;

        final amenities = <String>[
          ...parseList(iAmenities >= 0 ? row[iAmenities] : null),
          if (iWifi >= 0 && isYes(row[iWifi])) 'WiFi',
          if (iGym >= 0 && isYes(row[iGym])) 'Gym',
          if (iStudyRoom >= 0 && isYes(row[iStudyRoom])) 'Study Room',
          if (iCafeteria >= 0 && isYes(row[iCafeteria])) 'Cafeteria',
          if (iLaundry >= 0 && isYes(row[iLaundry])) 'Laundry',
          if (iAc >= 0 && isYes(row[iAc])) 'AC',
          if (iGenerator >= 0 && isYes(row[iGenerator])) 'Generator',
          if (iSecurityGuard >= 0 && isYes(row[iSecurityGuard])) 'Security Guard',
          if (iCctv >= 0 && isYes(row[iCctv])) 'CCTV',
          if (iHotWater >= 0 && isYes(row[iHotWater])) 'Hot Water',
          if (iLibrary >= 0 && isYes(row[iLibrary])) 'Library',
          if (iParking >= 0 && isYes(row[iParking])) 'Parking',
          if (iPrayerRoom >= 0 && isYes(row[iPrayerRoom])) 'Prayer Room',
          if (iCommonRoom >= 0 && isYes(row[iCommonRoom])) 'Common Room',
        ];

        final cleaned = amenities.map(prettyAmenity).where((e) => e.isNotEmpty).toSet().toList()..sort();
        amenitySet.addAll(cleaned);
        final roomTypes = parseList(iRoomTypes >= 0 ? row[iRoomTypes] : null)
            .map(prettyRoomType).toSet().toList()..sort();
        roomTypeSet.addAll(roomTypes);

        hostels.add(Hostel(
          id: row[iId].toString(), name: row[iName].toString(),
          type: iType >= 0 ? row[iType].toString() : '',
          area: iArea >= 0 ? row[iArea].toString() : '',
          city: iCity >= 0 ? row[iCity].toString() : '',
          address: iAddr >= 0 ? row[iAddr].toString() : '',
          lat: lat, lng: lng, amenities: cleaned,
          electricityIncluded: iElectricity >= 0 && isYes(row[iElectricity]),
          mealIncluded: iMeal >= 0 && isYes(row[iMeal]),
          foodType: iFoodType >= 0 ? row[iFoodType].toString().trim().isEmpty ? null : row[iFoodType].toString() : null,
          foodRating: iFoodRating >= 0 ? parseDouble(row[iFoodRating]) : null,
          roomTypesAvailable: roomTypes,
          overallRating: iRating >= 0 ? parseDouble(row[iRating]) : null,
          totalReviews: iReviews >= 0 ? parseInt(row[iReviews]) : null,
          singlePrice: iSingle >= 0 ? parseInt(row[iSingle]) : null,
          doublePrice: iDouble >= 0 ? parseInt(row[iDouble]) : null,
          dormPrice: iDorm >= 0 ? parseInt(row[iDorm]) : null,
        ));
      }

      setState(() {
        _allHostels = hostels;
        _amenityOptions = amenitySet.toList()..sort();
        _roomTypeOptions = roomTypeSet.toList()..sort();
        _loadingDataset = false;
      });
      _applyFilters();
    } catch (_) {
      setState(() { _loadingDataset = false; _statusText = 'Failed to load hostel dataset.'; });
    }
  }

  // ─── Filters ──────────────────────────────────────────────────────────────

  double _deg2rad(double d) => d * (pi / 180.0);

  double _distanceKm(LatLng a, LatLng b) {
    const r = 6371.0;
    final dLat = _deg2rad(b.latitude - a.latitude);
    final dLon = _deg2rad(b.longitude - a.longitude);
    final h = sin(dLat / 2) * sin(dLat / 2) +
        sin(dLon / 2) * sin(dLon / 2) * cos(_deg2rad(a.latitude)) * cos(_deg2rad(b.latitude));
    return 2 * r * atan2(sqrt(h), sqrt(1 - h));
  }

  void _applyFilters() {
    if (_loadingDataset) return;
    final filtered = _allHostels.where((h) {
      if (_distanceKm(_currentCenter, h.point) > _radiusKm) return false;
      if (_genderFilter == GenderFilter.girls && !h.type.toLowerCase().contains('girls')) return false;
      if (_genderFilter == GenderFilter.boys && !h.type.toLowerCase().contains('boys')) return false;
      if (_budgetFilterOn) {
        final p = h.minPrice;
        if (p == null) { if (!_includeUnknownPrice) return false; }
        else if (p < _budgetMin || p > _budgetMax) return false;
      }
      if (_selectedAmenities.isNotEmpty && !_selectedAmenities.every((a) => h.hasAmenity(a))) return false;
      if (_requireElectricityIncluded && !h.electricityIncluded) return false;
      if (_requireMealsIncluded && !h.mealIncluded) return false;
      if (_foodTypeFilter != FoodTypeFilter.all) {
        final ft = (h.foodType ?? '').trim().toLowerCase();
        if (_foodTypeFilter == FoodTypeFilter.veg && ft != 'veg') return false;
        if (_foodTypeFilter == FoodTypeFilter.nonVeg && ft != 'non-veg') return false;
        if (_foodTypeFilter == FoodTypeFilter.both && ft != 'both') return false;
      }
      if (_foodRatingFilterOn && (h.foodRating == null || h.foodRating! < _minFoodRating)) return false;
      if (_selectedRoomTypes.isNotEmpty && !_selectedRoomTypes.any((rt) => h.hasRoomType(rt))) return false;
      if (_hostelRatingFilterOn && (h.overallRating == null || h.overallRating! < _minHostelRating)) return false;
      if (_reviewsFilterOn && (h.totalReviews == null || h.totalReviews! < _minReviews)) return false;
      return true;
    }).toList();

    filtered.sort((a, b) => _distanceKm(_currentCenter, a.point).compareTo(_distanceKm(_currentCenter, b.point)));

    setState(() {
      _filteredHostels = filtered;
      _statusText = filtered.isEmpty
          ? 'No hostels match filters within ${_radiusKm.toStringAsFixed(1)} km.'
          : null;
    });
  }

  Future<void> _goToMyLocation() async {
    setState(() { _loading = true; _statusText = 'Getting your location...'; });
    final pos = await LocationService.getCurrentPosition();
    if (!mounted) return;
    if (pos == null) {
      setState(() { _loading = false; _locationEnabled = false; _statusText = 'Location unavailable. Tap Enable to try again.'; });
      return;
    }
    final here = LatLng(pos.latitude, pos.longitude);
    setState(() { _loading = false; _locationEnabled = true; _currentCenter = here; _zoom = 15; _searchMarker = null; _statusText = null; });
    _mapController.move(_currentCenter, _zoom);
    _applyFilters();
  }

  void _openAiRecommendationPage() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => const AiRecommendationScreen(),
      ),
    );
  }

  void _openChatbotPage() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => const ChatbotScreen(),
      ),
    );
  }

  // ─── Search ───────────────────────────────────────────────────────────────

  void _onSearchTextChanged() {
    if (_ignoreNextTextChange) { _ignoreNextTextChange = false; return; }
    final q = _searchController.text.trim();
    if (q.isEmpty) { setState(() { _suggestions = []; _showSuggestions = false; }); return; }
    _suggestDebounce?.cancel();
    _suggestDebounce = Timer(const Duration(milliseconds: 300), () => _fetchSuggestions(q));
  }

  Future<void> _fetchSuggestions(String query) async {
    setState(() { _loadingSuggestions = true; _showSuggestions = true; });
    final uri = Uri.parse('https://nominatim.openstreetmap.org/search?q=${Uri.encodeComponent(query)}&format=json&addressdetails=1&limit=7');
    try {
      final res = await http.get(uri, headers: {'User-Agent': 'staybuddy_student_app'});
      if (res.statusCode != 200) { setState(() { _loadingSuggestions = false; _suggestions = []; }); return; }
      final data = jsonDecode(res.body);
      if (data is! List) { setState(() { _loadingSuggestions = false; _suggestions = []; }); return; }
      final list = <PlaceSuggestion>[];
      for (final item in data) {
        final lat = double.tryParse(item['lat']?.toString() ?? '');
        final lon = double.tryParse(item['lon']?.toString() ?? '');
        if (lat == null || lon == null) continue;
        final display = (item['display_name'] ?? '').toString();
        if (display.isEmpty) continue;
        final parts = display.split(',');
        list.add(PlaceSuggestion(
          title: parts.isNotEmpty ? parts.first.trim() : display.trim(),
          subtitle: parts.length > 1 ? parts.sublist(1).join(',').trim() : '',
          point: LatLng(lat, lon),
        ));
      }
      setState(() { _loadingSuggestions = false; _suggestions = list; });
    } catch (_) {
      setState(() { _loadingSuggestions = false; _suggestions = []; });
    }
  }

  void _hideSuggestions() => setState(() { _showSuggestions = false; _suggestions = []; _loadingSuggestions = false; });

  void _selectSuggestion(PlaceSuggestion s) {
    _searchFocus.unfocus();
    _ignoreNextTextChange = true;
    setState(() {
      _searchController.text = s.title;
      _currentCenter = s.point;
      _zoom = 15;
      _searchMarker = Marker(point: s.point, width: 50, height: 50,
          child: const Icon(Icons.location_pin, size: 44, color: Colors.red));
    });
    _hideSuggestions();
    _mapController.move(_currentCenter, _zoom);
    _applyFilters();
  }

  void _zoomIn() { setState(() => _zoom = (_zoom + 1).clamp(3, 19)); _mapController.move(_currentCenter, _zoom); }
  void _zoomOut() { setState(() => _zoom = (_zoom - 1).clamp(3, 19)); _mapController.move(_currentCenter, _zoom); }

  int _countActiveFilters() {
    int c = 0;
    if (_budgetFilterOn) c++;
    if (_genderFilter != GenderFilter.all) c++;
    if (_selectedAmenities.isNotEmpty) c++;
    if (_requireElectricityIncluded) c++;
    if (_requireMealsIncluded) c++;
    if (_foodTypeFilter != FoodTypeFilter.all) c++;
    if (_foodRatingFilterOn) c++;
    if (_selectedRoomTypes.isNotEmpty) c++;
    if (_hostelRatingFilterOn) c++;
    if (_reviewsFilterOn) c++;
    return c;
  }

  // ─── Bottom Sheets ────────────────────────────────────────────────────────

  void _showStyledSheet(String title, IconData icon, Widget content) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (_) => _StyledSheet(title: title, icon: icon, child: content),
    );
  }

  void _openDistanceSheet() {
    double temp = _radiusKm;
    _showStyledSheet('Search by Distance', Icons.route_rounded, StatefulBuilder(
      builder: (ctx, set) => Column(mainAxisSize: MainAxisSize.min, children: [
        _sheetRow(
          left: 'Radius',
          right: '${temp.toStringAsFixed(1)} km',
        ),
        SliderTheme(
          data: _sliderTheme(context),
          child: Slider(value: temp, min: 1, max: 10, divisions: 18,
            label: '${temp.toStringAsFixed(1)} km',
            onChanged: (v) { set(() => temp = v); setState(() => _radiusKm = v); _applyFilters(); }),
        ),
        _applyBtn(() { Navigator.pop(ctx); _applyFilters(); }),
      ]),
    ));
  }

  void _openGenderSheet() {
    _showStyledSheet('Hostel Type', Icons.wc_rounded, Column(mainAxisSize: MainAxisSize.min, children: [
      Row(children: [
        Expanded(child: _choiceBtn('Girls', _genderFilter == GenderFilter.girls,
            () { setState(() => _genderFilter = GenderFilter.girls); Navigator.pop(context); _applyFilters(); })),
        const SizedBox(width: 10),
        Expanded(child: _choiceBtn('Boys', _genderFilter == GenderFilter.boys,
            () { setState(() => _genderFilter = GenderFilter.boys); Navigator.pop(context); _applyFilters(); })),
      ]),
      const SizedBox(height: 10),
      _outlineBtn('Show All', () { setState(() => _genderFilter = GenderFilter.all); Navigator.pop(context); _applyFilters(); }),
    ]));
  }

  void _openBudgetSheet() {
    RangeValues temp = RangeValues(_budgetMin.toDouble(), _budgetMax.toDouble());
    bool tempOn = _budgetFilterOn;
    bool tempInclude = _includeUnknownPrice;
    _showStyledSheet('Search by Budget', Icons.payments_outlined, StatefulBuilder(
      builder: (ctx, set) => Column(mainAxisSize: MainAxisSize.min, children: [
        _toggleRow('Enable budget filter', tempOn, (v) => set(() => tempOn = v)),
        _sheetRow(left: 'Range', right: 'PKR ${temp.start.toInt()} – ${temp.end.toInt()}'),
        SliderTheme(
          data: _sliderTheme(ctx),
          child: RangeSlider(values: temp, min: 5000, max: 100000, divisions: 95,
            labels: RangeLabels('${temp.start.toInt()}', '${temp.end.toInt()}'),
            onChanged: (v) => set(() => temp = v)),
        ),
        _toggleRow('Include hostels with unknown price', tempInclude, (v) => set(() => tempInclude = v)),
        const SizedBox(height: 6),
        _applyBtn(() {
          setState(() { _budgetFilterOn = tempOn; _budgetMin = temp.start.toInt(); _budgetMax = temp.end.toInt(); _includeUnknownPrice = tempInclude; });
          Navigator.pop(ctx); _applyFilters();
        }),
      ]),
    ));
  }

  void _openAmenitiesSheet() {
    final temp = <String>{..._selectedAmenities};
    showModalBottomSheet(
      context: context, backgroundColor: Colors.transparent, isScrollControlled: true,
      builder: (_) => _StyledSheet(
        title: 'Amenities', icon: Icons.checklist_rounded,
        child: StatefulBuilder(builder: (ctx, set) => Column(mainAxisSize: MainAxisSize.min, children: [
          Wrap(spacing: 8, runSpacing: 8, children: _amenityOptions.map((a) {
            final sel = temp.contains(a);
            return GestureDetector(
              onTap: () => set(() => sel ? temp.remove(a) : temp.add(a)),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 180),
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                decoration: BoxDecoration(
                  color: sel ? _teal : const Color(0xFFF5F3EE),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: sel ? _teal : _border),
                ),
                child: Text(a, style: GoogleFonts.dmSans(fontSize: 13, fontWeight: FontWeight.w500, color: sel ? Colors.white : _textPrimary)),
              ),
            );
          }).toList()),
          const SizedBox(height: 16),
          _applyBtn(() { setState(() { _selectedAmenities.clear(); _selectedAmenities.addAll(temp); }); Navigator.pop(ctx); _applyFilters(); }),
        ])),
      ),
    );
  }

  void _openStaySheet() {
    bool tempE = _requireElectricityIncluded, tempM = _requireMealsIncluded;
    _showStyledSheet('Stay Features', Icons.home_rounded, StatefulBuilder(
      builder: (ctx, set) => Column(mainAxisSize: MainAxisSize.min, children: [
        _toggleRow('Electricity included', tempE, (v) => set(() => tempE = v)),
        _toggleRow('Meals included', tempM, (v) => set(() => tempM = v)),
        const SizedBox(height: 6),
        _applyBtn(() { setState(() { _requireElectricityIncluded = tempE; _requireMealsIncluded = tempM; }); Navigator.pop(ctx); _applyFilters(); }),
      ]),
    ));
  }

  void _openFoodSheet() {
    FoodTypeFilter tempType = _foodTypeFilter;
    bool tempOn = _foodRatingFilterOn;
    double tempRating = _minFoodRating;
    _showStyledSheet('Food', Icons.restaurant_rounded, StatefulBuilder(
      builder: (ctx, set) => Column(mainAxisSize: MainAxisSize.min, children: [
        Wrap(spacing: 8, children: FoodTypeFilter.values.map((f) {
          final labels = {FoodTypeFilter.all: 'All', FoodTypeFilter.veg: 'Veg', FoodTypeFilter.nonVeg: 'Non-Veg', FoodTypeFilter.both: 'Both'};
          return GestureDetector(
            onTap: () => set(() => tempType = f),
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 180),
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 9),
              margin: const EdgeInsets.only(bottom: 6),
              decoration: BoxDecoration(color: tempType == f ? _teal : const Color(0xFFF5F3EE), borderRadius: BorderRadius.circular(20), border: Border.all(color: tempType == f ? _teal : _border)),
              child: Text(labels[f]!, style: GoogleFonts.dmSans(fontSize: 13, fontWeight: FontWeight.w500, color: tempType == f ? Colors.white : _textPrimary)),
            ),
          );
        }).toList()),
        const SizedBox(height: 8),
        _toggleRow('Min food rating', tempOn, (v) => set(() => tempOn = v)),
        SliderTheme(data: _sliderTheme(ctx), child: Slider(value: tempRating, min: 1, max: 5, divisions: 8,
          label: tempRating.toStringAsFixed(1), onChanged: (v) => set(() => tempRating = v))),
        _applyBtn(() { setState(() { _foodTypeFilter = tempType; _foodRatingFilterOn = tempOn; _minFoodRating = tempRating; }); Navigator.pop(ctx); _applyFilters(); }),
      ]),
    ));
  }

  void _openRoomTypeSheet() {
    final temp = <String>{..._selectedRoomTypes};
    _showStyledSheet('Room Type', Icons.bed_rounded, StatefulBuilder(
      builder: (ctx, set) => Column(mainAxisSize: MainAxisSize.min, children: [
        Wrap(spacing: 8, runSpacing: 8, children: _roomTypeOptions.map((rt) {
          final sel = temp.contains(rt);
          return GestureDetector(
            onTap: () => set(() => sel ? temp.remove(rt) : temp.add(rt)),
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 180),
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              decoration: BoxDecoration(color: sel ? _teal : const Color(0xFFF5F3EE), borderRadius: BorderRadius.circular(20), border: Border.all(color: sel ? _teal : _border)),
              child: Text(rt, style: GoogleFonts.dmSans(fontSize: 13, fontWeight: FontWeight.w500, color: sel ? Colors.white : _textPrimary)),
            ),
          );
        }).toList()),
        const SizedBox(height: 16),
        _applyBtn(() { setState(() { _selectedRoomTypes.clear(); _selectedRoomTypes.addAll(temp); }); Navigator.pop(ctx); _applyFilters(); }),
      ]),
    ));
  }

  void _openRatingSheet() {
    bool tempRatingOn = _hostelRatingFilterOn, tempReviewsOn = _reviewsFilterOn;
    double tempRating = _minHostelRating, tempReviews = _minReviews.toDouble();
    _showStyledSheet('Rating & Reviews', Icons.star_rounded, StatefulBuilder(
      builder: (ctx, set) => Column(mainAxisSize: MainAxisSize.min, children: [
        _toggleRow('Min hostel rating', tempRatingOn, (v) => set(() => tempRatingOn = v)),
        _sheetRow(left: 'Min rating', right: tempRating.toStringAsFixed(1)),
        SliderTheme(data: _sliderTheme(ctx), child: Slider(value: tempRating, min: 1, max: 5, divisions: 8, label: tempRating.toStringAsFixed(1), onChanged: (v) => set(() => tempRating = v))),
        _toggleRow('Min review count', tempReviewsOn, (v) => set(() => tempReviewsOn = v)),
        _sheetRow(left: 'Min reviews', right: tempReviews.toInt().toString()),
        SliderTheme(data: _sliderTheme(ctx), child: Slider(value: tempReviews, min: 0, max: 150, divisions: 30, label: tempReviews.toInt().toString(), onChanged: (v) => set(() => tempReviews = v))),
        _applyBtn(() { setState(() { _hostelRatingFilterOn = tempRatingOn; _minHostelRating = tempRating; _reviewsFilterOn = tempReviewsOn; _minReviews = tempReviews.toInt(); }); Navigator.pop(ctx); _applyFilters(); }),
      ]),
    ));
  }

  // ─── Sheet Helpers ────────────────────────────────────────────────────────

  SliderThemeData _sliderTheme(BuildContext ctx) => SliderTheme.of(ctx).copyWith(
    activeTrackColor: _teal, thumbColor: _teal, inactiveTrackColor: _border,
    overlayColor: _teal.withOpacity(0.1), trackHeight: 4,
    valueIndicatorColor: _teal,
    valueIndicatorTextStyle: GoogleFonts.dmSans(color: Colors.white, fontSize: 12),
  );

  Widget _sheetRow({required String left, required String right}) => Padding(
    padding: const EdgeInsets.symmetric(vertical: 4),
    child: Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
      Text(left, style: GoogleFonts.dmSans(fontSize: 13, color: _textMuted)),
      Text(right, style: GoogleFonts.dmSans(fontSize: 14, fontWeight: FontWeight.w600, color: _textPrimary)),
    ]),
  );

  Widget _toggleRow(String label, bool value, ValueChanged<bool> onChanged) => Padding(
    padding: const EdgeInsets.symmetric(vertical: 2),
    child: Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
      Text(label, style: GoogleFonts.dmSans(fontSize: 14, color: _textPrimary)),
      Switch(value: value, onChanged: onChanged, activeColor: _teal),
    ]),
  );

  Widget _applyBtn(VoidCallback onTap) => SizedBox(
    width: double.infinity, height: 50,
    child: ElevatedButton(
      onPressed: onTap,
      style: ElevatedButton.styleFrom(backgroundColor: _teal, foregroundColor: Colors.white, elevation: 0, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16))),
      child: Text('Apply', style: GoogleFonts.dmSans(fontSize: 15, fontWeight: FontWeight.w600)),
    ),
  );

  Widget _choiceBtn(String label, bool active, VoidCallback onTap) => GestureDetector(
    onTap: onTap,
    child: AnimatedContainer(
      duration: const Duration(milliseconds: 200),
      height: 50, alignment: Alignment.center,
      decoration: BoxDecoration(color: active ? _teal : const Color(0xFFF5F3EE), borderRadius: BorderRadius.circular(14), border: Border.all(color: active ? _teal : _border)),
      child: Text(label, style: GoogleFonts.dmSans(fontSize: 14, fontWeight: FontWeight.w600, color: active ? Colors.white : _textPrimary)),
    ),
  );

  Widget _outlineBtn(String label, VoidCallback onTap) => SizedBox(
    width: double.infinity, height: 46,
    child: OutlinedButton(
      onPressed: onTap,
      style: OutlinedButton.styleFrom(foregroundColor: _teal, side: const BorderSide(color: _teal), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14))),
      child: Text(label, style: GoogleFonts.dmSans(fontSize: 14, fontWeight: FontWeight.w600)),
    ),
  );

  // ─── Hostel Preview ───────────────────────────────────────────────────────

  void _selectHostel(Hostel h) {
    setState(() => _selectedHostel = h);
    _previewAnim.forward(from: 0);
  }

  void _dismissPreview() {
    _previewAnim.reverse().then((_) => setState(() => _selectedHostel = null));
  }

  // ─── Build ────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final markers = <Marker>[
      // My location
      Marker(
        point: _currentCenter, width: 52, height: 52,
        child: _MyLocationMarker(),
      ),
      if (_searchMarker != null) _searchMarker!,
      ..._filteredHostels.map((h) {
        final isSelected = _selectedHostel?.id == h.id;
        return Marker(
          point: h.point, width: 52, height: 52,
          child: GestureDetector(
            onTap: () => isSelected ? _dismissPreview() : _selectHostel(h),
            child: _HostelMarker(hostel: h, isSelected: isSelected),
          ),
        );
      }),
    ];

    return ResponsiveShell(
      child: GestureDetector(
      onTap: () { FocusScope.of(context).unfocus(); _hideSuggestions(); },
      child: Scaffold(
        backgroundColor: Colors.black,
        body: Stack(
          children: [
            // ── Map ────────────────────────────────────────────────────────
            FlutterMap(
              mapController: _mapController,
              options: MapOptions(
                initialCenter: _currentCenter, initialZoom: _zoom,
                onTap: (_, __) { _hideSuggestions(); if (_selectedHostel != null) _dismissPreview(); },
              ),
              children: [
                TileLayer(urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png', userAgentPackageName: 'staybuddy_student'),
                MarkerLayer(markers: markers),
              ],
            ),

            // ── Top Search Bar ─────────────────────────────────────────────
            Positioned(
              left: 14, right: 14, top: MediaQuery.of(context).padding.top + 10,
              child: Column(children: [
                _SearchBar(
                  controller: _searchController, focusNode: _searchFocus,
                  onBack: () => Navigator.pushReplacementNamed(context, Routes.locationAccess),
                  onSearch: () { if (_suggestions.isNotEmpty) _selectSuggestion(_suggestions.first); },
                ),
                if (_showSuggestions) ...[
                  const SizedBox(height: 8),
                  _SuggestionsPanel(
                    loading: _loadingSuggestions,
                    suggestions: _suggestions,
                    currentCenter: _currentCenter,
                    onSelect: _selectSuggestion,
                    distanceKm: _distanceKm,
                  ),
                ],
              ]),
            ),

            // ── Location disabled banner ───────────────────────────────────
            if (!_locationEnabled)
              Positioned(
                left: 16, right: 16, bottom: 230,
                child: _LocationBanner(onEnable: _goToMyLocation),
              ),

            // ── Status toast ──────────────────────────────────────────────
            if (_loading || _loadingDataset || _statusText != null)
              Positioned(
                left: 60, right: 60, bottom: 240,
                child: _StatusToast(
                  loading: _loading || _loadingDataset,
                  text: _statusText ?? (_loadingDataset ? 'Loading dataset...' : 'Working...'),
                ),
              ),

            // ── Hostel preview card ────────────────────────────────────────
            if (_selectedHostel != null)
              Positioned(
                left: 14, right: 80, bottom: 210,
                child: AnimatedBuilder(
                  animation: _previewSlide,
                  builder: (_, child) => Transform.translate(
                    offset: Offset(0, 30 * (1 - _previewSlide.value)),
                    child: Opacity(opacity: _previewSlide.value, child: child),
                  ),
                  child: _HostelPreviewCard(
                    hostel: _selectedHostel!,
                    distanceKm: _distanceKm(_currentCenter, _selectedHostel!.point),
                    onDismiss: _dismissPreview,
                  ),
                ),
              ),

            // ── FABs ──────────────────────────────────────────────────────
            Positioned(
              right: 14, bottom: 215,
              child: ScaleTransition(
                scale: _fabScale,
                child: Column(children: [
                  _MapFab(icon: Icons.add_rounded, onTap: _zoomIn),
                  const SizedBox(height: 8),
                  _MapFab(icon: Icons.remove_rounded, onTap: _zoomOut),
                  const SizedBox(height: 8),
                  _MapFab(icon: Icons.my_location_rounded, onTap: _goToMyLocation, highlight: _locationEnabled),
                  const SizedBox(height: 8),
                  _MapFab(icon: Icons.chat_bubble_outline_rounded, onTap: _openChatbotPage),
                ]),
              ),
            ),

            // ── Bottom panel ──────────────────────────────────────────────
            Positioned(
              left: 0, right: 0, bottom: 0,
              child: SlideTransition(
                position: Tween<Offset>(begin: const Offset(0, 1), end: Offset.zero)
                    .animate(_bottomSheetSlide),
                child: _BottomPanel(
                  hostelCount: _filteredHostels.length,
                  radiusKm: _radiusKm,
                  activeFilterCount: _countActiveFilters(),
                  onDistance: _openDistanceSheet,
                  onAiRecommend: _openAiRecommendationPage,
                  onHostelType: _openGenderSheet,
                  onBudget: _openBudgetSheet,
                  amenityCount: _selectedAmenities.length,
                  onAmenities: _openAmenitiesSheet,
                  onStay: _openStaySheet,
                  onFood: _openFoodSheet,
                  foodActive: _foodTypeFilter != FoodTypeFilter.all || _foodRatingFilterOn,
                  roomTypeCount: _selectedRoomTypes.length,
                  onRoomType: _openRoomTypeSheet,
                  onRating: _openRatingSheet,
                  ratingActive: _hostelRatingFilterOn || _reviewsFilterOn,
                ),
              ),
            ),
          ],
        ),
      ),
    ),
    );
  }
}

// ─── Sub-widgets ─────────────────────────────────────────────────────────────

class _MyLocationMarker extends StatefulWidget {
  @override
  State<_MyLocationMarker> createState() => _MyLocationMarkerState();
}

class _MyLocationMarkerState extends State<_MyLocationMarker>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;
  late final Animation<double> _pulse;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(vsync: this, duration: const Duration(seconds: 2))..repeat();
    _pulse = CurvedAnimation(parent: _ctrl, curve: Curves.easeOut);
  }

  @override
  void dispose() { _ctrl.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _pulse,
      builder: (_, child) => Stack(alignment: Alignment.center, children: [
        Container(
          width: 44 + 16 * _pulse.value,
          height: 44 + 16 * _pulse.value,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: Colors.blue.withOpacity(0.15 * (1 - _pulse.value)),
          ),
        ),
        child!,
      ]),
      child: Container(
        width: 22, height: 22,
        decoration: BoxDecoration(
          color: Colors.blue, shape: BoxShape.circle,
          border: Border.all(color: Colors.white, width: 2.5),
          boxShadow: [BoxShadow(color: Colors.blue.withOpacity(0.4), blurRadius: 8)],
        ),
      ),
    );
  }
}

class _HostelMarker extends StatelessWidget {
  final Hostel hostel;
  final bool isSelected;
  const _HostelMarker({required this.hostel, required this.isSelected});

  @override
  Widget build(BuildContext context) {
    return AnimatedScale(
      scale: isSelected ? 1.3 : 1.0,
      duration: const Duration(milliseconds: 200),
      curve: Curves.elasticOut,
      child: Container(
        width: 38, height: 38,
        decoration: BoxDecoration(
          color: isSelected ? _tealDark : _teal,
          shape: BoxShape.circle,
          border: Border.all(color: Colors.white, width: isSelected ? 2.5 : 2),
          boxShadow: [BoxShadow(color: _teal.withOpacity(isSelected ? 0.5 : 0.3), blurRadius: isSelected ? 12 : 6, offset: const Offset(0, 3))],
        ),
        child: Icon(Icons.home_rounded, color: Colors.white, size: isSelected ? 20 : 18),
      ),
    );
  }
}

class _SearchBar extends StatelessWidget {
  final TextEditingController controller;
  final FocusNode focusNode;
  final VoidCallback onBack, onSearch;
  const _SearchBar({required this.controller, required this.focusNode, required this.onBack, required this.onSearch});

  @override
  Widget build(BuildContext context) {
    return Row(children: [
      // Back button
      GestureDetector(
        onTap: onBack,
        child: Container(
          width: 46, height: 46,
          decoration: BoxDecoration(color: _teal, shape: BoxShape.circle,
            boxShadow: [BoxShadow(color: _teal.withOpacity(0.35), blurRadius: 12, offset: const Offset(0, 4))]),
          child: const Icon(Icons.arrow_back_rounded, color: Colors.white, size: 20),
        ),
      ),
      const SizedBox(width: 10),
      // Search field
      Expanded(
        child: Container(
          height: 46,
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(23),
            boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.12), blurRadius: 20, offset: const Offset(0, 4))],
          ),
          child: Row(children: [
            const SizedBox(width: 14),
            Icon(Icons.search_rounded, color: _textMuted, size: 20),
            const SizedBox(width: 8),
            Expanded(
              child: TextField(
                controller: controller, focusNode: focusNode,
                style: GoogleFonts.dmSans(fontSize: 14, color: _textPrimary),
                decoration: InputDecoration(
                  border: InputBorder.none,
                  hintText: 'Search by city, university or area',
                  hintStyle: GoogleFonts.dmSans(fontSize: 13, color: _textMuted),
                ),
              ),
            ),
            const SizedBox(width: 8),
          ]),
        ),
      ),
      const SizedBox(width: 10),
      // Search button
      GestureDetector(
        onTap: onSearch,
        child: Container(
          width: 46, height: 46,
          decoration: BoxDecoration(color: _teal, shape: BoxShape.circle,
            boxShadow: [BoxShadow(color: _teal.withOpacity(0.35), blurRadius: 12, offset: const Offset(0, 4))]),
          child: const Icon(Icons.search_rounded, color: Colors.white, size: 20),
        ),
      ),
    ]);
  }
}

class _SuggestionsPanel extends StatelessWidget {
  final bool loading;
  final List<PlaceSuggestion> suggestions;
  final LatLng currentCenter;
  final ValueChanged<PlaceSuggestion> onSelect;
  final double Function(LatLng, LatLng) distanceKm;

  const _SuggestionsPanel({required this.loading, required this.suggestions, required this.currentCenter, required this.onSelect, required this.distanceKm});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white, borderRadius: BorderRadius.circular(20),
        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.12), blurRadius: 24, offset: const Offset(0, 8))],
      ),
      child: loading
          ? Padding(padding: const EdgeInsets.all(16),
              child: Row(children: [
                SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2, color: _teal)),
                const SizedBox(width: 12),
                Text('Searching...', style: GoogleFonts.dmSans(color: _textMuted, fontSize: 13)),
              ]))
          : suggestions.isEmpty
              ? Padding(padding: const EdgeInsets.all(16),
                  child: Text('No results found', style: GoogleFonts.dmSans(color: _textMuted, fontSize: 13)))
              : ListView.separated(
                  shrinkWrap: true, physics: const NeverScrollableScrollPhysics(),
                  itemCount: suggestions.length,
                  separatorBuilder: (_, __) => Divider(height: 0.5, color: _border),
                  itemBuilder: (_, i) {
                    final s = suggestions[i];
                    final d = distanceKm(currentCenter, s.point);
                    return ListTile(
                      dense: true,
                      leading: Container(width: 32, height: 32,
                        decoration: BoxDecoration(color: _teal.withOpacity(0.1), shape: BoxShape.circle),
                        child: const Icon(Icons.location_on_rounded, size: 16, color: _teal)),
                      title: Text(s.title, style: GoogleFonts.dmSans(fontSize: 13, fontWeight: FontWeight.w600, color: _textPrimary)),
                      subtitle: Text(s.subtitle, maxLines: 1, overflow: TextOverflow.ellipsis,
                        style: GoogleFonts.dmSans(fontSize: 11, color: _textMuted)),
                      trailing: Text('${d.toStringAsFixed(1)} km',
                        style: GoogleFonts.dmSans(fontSize: 11, color: _teal, fontWeight: FontWeight.w600)),
                      onTap: () => onSelect(s),
                    );
                  },
                ),
    );
  }
}

class _LocationBanner extends StatelessWidget {
  final VoidCallback onEnable;
  const _LocationBanner({required this.onEnable});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: Colors.white, borderRadius: BorderRadius.circular(16),
        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.1), blurRadius: 16)],
        border: Border.all(color: _border, width: 0.5),
      ),
      child: Row(children: [
        Container(width: 36, height: 36, decoration: BoxDecoration(color: Colors.orange.shade50, shape: BoxShape.circle),
          child: Icon(Icons.location_off_rounded, size: 18, color: Colors.orange.shade700)),
        const SizedBox(width: 10),
        Expanded(child: Text('Location is off. Enable to see nearby hostels.',
          style: GoogleFonts.dmSans(fontSize: 12, color: _textPrimary))),
        GestureDetector(onTap: onEnable,
          child: Container(padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(color: _teal, borderRadius: BorderRadius.circular(10)),
            child: Text('Enable', style: GoogleFonts.dmSans(fontSize: 12, fontWeight: FontWeight.w600, color: Colors.white)))),
      ]),
    );
  }
}

class _StatusToast extends StatelessWidget {
  final bool loading;
  final String text;
  const _StatusToast({required this.loading, required this.text});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      decoration: BoxDecoration(
        color: _tealDark.withOpacity(0.92), borderRadius: BorderRadius.circular(24),
        boxShadow: [BoxShadow(color: _teal.withOpacity(0.3), blurRadius: 12)],
      ),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        if (loading) ...[
          SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white)),
          const SizedBox(width: 10),
        ],
        Flexible(child: Text(text, textAlign: TextAlign.center,
          style: GoogleFonts.dmSans(fontSize: 12, color: Colors.white, fontWeight: FontWeight.w500))),
      ]),
    );
  }
}

class _HostelPreviewCard extends StatelessWidget {
  final Hostel hostel;
  final double distanceKm;
  final VoidCallback onDismiss;
  const _HostelPreviewCard({required this.hostel, required this.distanceKm, required this.onDismiss});

  @override
  Widget build(BuildContext context) {
    final price = hostel.minPrice;
    final isGirls = hostel.type.toLowerCase().contains('girl') ||
        hostel.name.toLowerCase().contains('girl') ||
        hostel.name.toLowerCase().contains('female');
    final typeColor = isGirls ? const Color(0xFFE91E8C) : const Color(0xFF185FA5);
    final typeLabel = isGirls ? 'Girls' : 'Boys';

    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
        boxShadow: [BoxShadow(
            color: Colors.black.withOpacity(0.18),
            blurRadius: 28, offset: const Offset(0, 10))],
        border: Border.all(color: _border, width: 0.5),
      ),
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        // Top row - hostel info
        Padding(
          padding: const EdgeInsets.fromLTRB(14, 14, 14, 10),
          child: Row(children: [
            Container(
              width: 52, height: 52,
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [_teal, _tealLight],
                  begin: Alignment.topLeft, end: Alignment.bottomRight),
                borderRadius: BorderRadius.circular(14)),
              child: const Icon(Icons.home_rounded, color: Colors.white, size: 26)),
            const SizedBox(width: 12),
            Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Row(children: [
                Expanded(child: Text(hostel.name,
                    style: GoogleFonts.dmSans(
                        fontSize: 15, fontWeight: FontWeight.w700,
                        color: _textPrimary),
                    maxLines: 1, overflow: TextOverflow.ellipsis)),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
                  decoration: BoxDecoration(
                    color: typeColor.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(6)),
                  child: Text(typeLabel,
                      style: GoogleFonts.dmSans(
                          fontSize: 10, fontWeight: FontWeight.w600,
                          color: typeColor))),
              ]),
              const SizedBox(height: 3),
              Row(children: [
                const Icon(Icons.location_on_rounded, size: 12, color: _textMuted),
                const SizedBox(width: 3),
                Expanded(child: Text(
                  '${hostel.area.isNotEmpty ? hostel.area : hostel.city} · ${distanceKm.toStringAsFixed(1)} km',
                  style: GoogleFonts.dmSans(fontSize: 11, color: _textMuted),
                  overflow: TextOverflow.ellipsis)),
              ]),
              const SizedBox(height: 5),
              Row(children: [
                if (price != null) ...[
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                    decoration: BoxDecoration(
                      color: _teal.withOpacity(0.08),
                      borderRadius: BorderRadius.circular(6)),
                    child: Text('PKR $price/mo',
                        style: GoogleFonts.dmSans(
                            fontSize: 11, fontWeight: FontWeight.w600,
                            color: _teal))),
                  const SizedBox(width: 8),
                ],
                if (hostel.overallRating != null)
                  Row(children: [
                    Icon(Icons.star_rounded, size: 13, color: Colors.amber.shade600),
                    const SizedBox(width: 2),
                    Text(hostel.overallRating!.toStringAsFixed(1),
                        style: GoogleFonts.dmSans(
                            fontSize: 11, fontWeight: FontWeight.w600,
                            color: Colors.amber.shade700)),
                  ]),
              ]),
            ])),
            GestureDetector(
              onTap: onDismiss,
              child: Container(
                width: 28, height: 28,
                decoration: const BoxDecoration(
                    color: Color(0xFFF5F3EE), shape: BoxShape.circle),
                child: const Icon(Icons.close_rounded,
                    size: 16, color: _textMuted))),
          ]),
        ),

        // Divider
        Container(height: 0.5, color: _border),

        // Bottom row - Book Now button
        Padding(
          padding: const EdgeInsets.fromLTRB(14, 10, 14, 14),
          child: Row(children: [
            // View details
            Expanded(
              flex: 1,
              child: GestureDetector(
                onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (_) => HostelDetailScreen(
                      id: int.tryParse(hostel.id) ?? hostel.id.hashCode,
                    ),
                  ),
                ),
                child: Container(
                  height: 42,
                  decoration: BoxDecoration(
                    color: const Color(0xFFF5F3EE),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: _border, width: 0.5)),
                  child: Row(mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                    const Icon(Icons.info_outline_rounded,
                        size: 16, color: _textMuted),
                    const SizedBox(width: 5),
                    Text('Details',
                        style: GoogleFonts.dmSans(
                            fontSize: 13, fontWeight: FontWeight.w600,
                            color: _textMuted)),
                  ]),
                ),
              ),
            ),
            const SizedBox(width: 10),
            // Book Now — main CTA
            Expanded(
              flex: 2,
              child: GestureDetector(
                onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (_) => StudentBookingInfoScreen(
                      hostelId: int.tryParse(hostel.id) ?? hostel.id.hashCode,
                      hostelName: hostel.name,
                      hostelType: hostel.type,
                      city: hostel.city,
                      address: hostel.address,
                      pricePerMonth: price ?? 0,
                      distanceKm: distanceKm,
                      rating: hostel.overallRating,
                    ),
                  ),
                ),
                child: Container(
                  height: 42,
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(
                      colors: [_teal, _tealLight],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight),
                    borderRadius: BorderRadius.circular(12),
                    boxShadow: [BoxShadow(
                        color: _teal.withOpacity(0.35),
                        blurRadius: 10, offset: const Offset(0, 4))]),
                  child: Row(mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                    const Icon(Icons.hotel_rounded,
                        size: 18, color: Colors.white),
                    const SizedBox(width: 7),
                    Text('Book Now',
                        style: GoogleFonts.dmSans(
                            fontSize: 14, fontWeight: FontWeight.w700,
                            color: Colors.white)),
                  ]),
                ),
              ),
            ),
          ]),
        ),
      ]),
    );
  }
}

class _MapFab extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;
  final bool highlight;
  const _MapFab({required this.icon, required this.onTap, this.highlight = false});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 44, height: 44,
        decoration: BoxDecoration(
          color: highlight ? _teal : Colors.white,
          shape: BoxShape.circle,
          border: highlight ? null : Border.all(color: _border, width: 0.5),
          boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.1), blurRadius: 10, offset: const Offset(0, 3))],
        ),
        child: Icon(icon, size: 20, color: highlight ? Colors.white : _textPrimary),
      ),
    );
  }
}

class _BottomPanel extends StatelessWidget {
  final int hostelCount, activeFilterCount, amenityCount, roomTypeCount;
  final double radiusKm;
  final bool foodActive, ratingActive;
  final VoidCallback onDistance, onAiRecommend, onHostelType, onBudget,
      onAmenities, onStay, onFood, onRoomType, onRating;

  const _BottomPanel({
    required this.hostelCount, required this.activeFilterCount,
    required this.radiusKm, required this.amenityCount,
    required this.roomTypeCount, required this.foodActive,
    required this.ratingActive, required this.onDistance,
    required this.onAiRecommend, required this.onHostelType,
    required this.onBudget, required this.onAmenities,
    required this.onStay, required this.onFood,
    required this.onRoomType, required this.onRating,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.12), blurRadius: 24, offset: const Offset(0, -4))],
      ),
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        // Handle
        Center(child: Container(width: 40, height: 4, margin: const EdgeInsets.only(top: 12, bottom: 14),
          decoration: BoxDecoration(color: _border, borderRadius: BorderRadius.circular(2)))),

        // Header row
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20),
          child: Row(children: [
            Row(children: [
              Container(width: 30, height: 30,
                decoration: BoxDecoration(color: _teal.withOpacity(0.1), borderRadius: BorderRadius.circular(8)),
                child: const Icon(Icons.home_work_rounded, size: 16, color: _teal)),
              const SizedBox(width: 10),
              Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text('Discover Hostels', style: GoogleFonts.playfairDisplay(fontSize: 16, fontWeight: FontWeight.w700, color: _textPrimary)),
                Text('$hostelCount found · ${radiusKm.toStringAsFixed(1)} km radius',
                  style: GoogleFonts.dmSans(fontSize: 11, color: _textMuted)),
              ]),
            ]),
            const Spacer(),
            if (activeFilterCount > 0)
              Container(padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(color: _teal, borderRadius: BorderRadius.circular(12)),
                child: Text('$activeFilterCount active', style: GoogleFonts.dmSans(fontSize: 11, fontWeight: FontWeight.w600, color: Colors.white))),
          ]),
        ),

        const SizedBox(height: 14),

        // Filter chips
        SizedBox(height: 40, child: ListView(
          scrollDirection: Axis.horizontal, padding: const EdgeInsets.symmetric(horizontal: 16),
          children: [
            _FilterChip(label: 'Distance', icon: Icons.route_rounded, onTap: onDistance),
            _FilterChip(label: 'AI Picks', icon: Icons.auto_awesome_rounded, onTap: onAiRecommend, accent: true),
            _FilterChip(label: 'Type', icon: Icons.wc_rounded, onTap: onHostelType),
            _FilterChip(label: 'Budget', icon: Icons.payments_outlined, onTap: onBudget),
            _FilterChip(label: amenityCount > 0 ? 'Amenities ($amenityCount)' : 'Amenities', icon: Icons.checklist_rounded, onTap: onAmenities, active: amenityCount > 0),
            _FilterChip(label: 'Stay', icon: Icons.home_rounded, onTap: onStay),
            _FilterChip(label: 'Food', icon: Icons.restaurant_rounded, onTap: onFood, active: foodActive),
            _FilterChip(label: roomTypeCount > 0 ? 'Rooms ($roomTypeCount)' : 'Rooms', icon: Icons.bed_rounded, onTap: onRoomType, active: roomTypeCount > 0),
            _FilterChip(label: 'Rating', icon: Icons.star_rounded, onTap: onRating, active: ratingActive),
          ],
        )),

        SizedBox(height: MediaQuery.of(context).padding.bottom + 14),
      ]),
    );
  }
}

class _FilterChip extends StatelessWidget {
  final String label;
  final IconData icon;
  final VoidCallback onTap;
  final bool active, accent;

  const _FilterChip({required this.label, required this.icon, required this.onTap, this.active = false, this.accent = false});

  @override
  Widget build(BuildContext context) {
    final bg = accent ? _tealLight : (active ? _teal : const Color(0xFFF5F3EE));
    final fg = (accent || active) ? Colors.white : _textPrimary;
    final borderColor = (active || accent) ? Colors.transparent : _border;

    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        margin: const EdgeInsets.only(right: 8),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 0),
        decoration: BoxDecoration(
          color: bg, borderRadius: BorderRadius.circular(20),
          border: Border.all(color: borderColor, width: 0.5),
          boxShadow: (active || accent) ? [BoxShadow(color: _teal.withOpacity(0.25), blurRadius: 8, offset: const Offset(0, 2))] : null,
        ),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          Icon(icon, size: 14, color: fg),
          const SizedBox(width: 5),
          Text(label, style: GoogleFonts.dmSans(fontSize: 12, fontWeight: FontWeight.w600, color: fg)),
        ]),
      ),
    );
  }
}

// ─── Styled Bottom Sheet ──────────────────────────────────────────────────────

class _StyledSheet extends StatelessWidget {
  final String title;
  final IconData icon;
  final Widget child;
  const _StyledSheet({required this.title, required this.icon, required this.child});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
      ),
      child: SafeArea(
        top: false,
        child: Padding(
          padding: EdgeInsets.fromLTRB(20, 0, 20, MediaQuery.of(context).viewInsets.bottom + 16),
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            // Handle
            Center(child: Container(width: 40, height: 4, margin: const EdgeInsets.symmetric(vertical: 14),
              decoration: BoxDecoration(color: _border, borderRadius: BorderRadius.circular(2)))),
            // Title
            Row(mainAxisAlignment: MainAxisAlignment.center, children: [
              Container(width: 36, height: 36, decoration: BoxDecoration(color: _teal.withOpacity(0.1), shape: BoxShape.circle),
                child: Icon(icon, size: 18, color: _teal)),
              const SizedBox(width: 10),
              Text(title, style: GoogleFonts.playfairDisplay(fontSize: 18, fontWeight: FontWeight.w700, color: _textPrimary)),
            ]),
            const SizedBox(height: 20),
            child,
          ]),
        ),
      ),
    );
  }
}